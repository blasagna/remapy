"""Shared wire protocol for the Feather Sense sensor stream.

The board streams only **raw** sensor samples (acceleration, angular rate,
magnetic field, battery); anything derivable from those is reconstructed on the
host (see ``motion.py``) rather than sent, so the device's loop budget goes to
sampling. See :data:`MSG_INFO` for the live types.

One sensor sample = one *frame*. A frame is a Type-Length-Value (TLV) record,
COBS-encoded, terminated by a single ``0x00`` delimiter::

    frame  = cobs_encode(payload) + b"\\x00"
    payload = [ type:u8 ][ length:u8 ][ value: `length` bytes ]
    value   = [ timestamp_ms:u32 ][ data: N x int32 ]   (little-endian)

**No floating point is put on the wire.** Sensor values are transmitted as
scaled fixed-point ``int32`` (e.g. milli-m/s^2). The scale factor for each field
lives in :data:`SCALES`; the device converts SI -> int with :func:`to_raw`, and
the host converts back int -> SI (see ``read_stream.py``). This keeps frames
compact and the numeric format explicit.

COBS (Consistent Overhead Byte Stuffing) removes every ``0x00`` byte from the
payload, so the lone trailing ``0x00`` unambiguously marks a frame boundary.
The host splits the byte stream on ``0x00`` and decodes each chunk, which makes
framing self-synchronising: a partial or corrupt frame is dropped and the next
delimiter resyncs the stream. Because each sample is framed independently,
sensors may be sampled at completely different rates.

This module is pure ``struct`` + ``bytes`` so the exact same file runs under
CircuitPython (on the board, used for :func:`encode`) and CPython (on the host,
used for :class:`FrameDecoder`). Keep it dependency-free.
"""

import struct

# --- Message types -----------------------------------------------------------
# Data values are scaled int32 (see SCALES); the battery's usb flag is a u8.
#
# Codes are assigned densely and carry no compatibility guarantee: the board and
# the host ship from this one file, so a renumbering is resolved by reflashing.
# (Recordings are unaffected — /feather groups are keyed by stream *name*, not by
# type code.) The env/altitude types of the removed BMP280/SHT31-D sensing are
# gone rather than reserved; if you re-add a stream, append the next free code.
MSG_ACCEL = 0x01     # 3 x i32: x, y, z            (m/s^2  * 1000)
MSG_GYRO = 0x02      # 3 x i32: x, y, z            (rad/s  * 10000)
MSG_MAG = 0x03       # 3 x i32: x, y, z            (uT     * 100)
MSG_BATTERY = 0x04   # 2 x i32 + 1 x u8: voltage(mV), percent(*100), usb_connected
MSG_ERROR = 0x05     # u8 source-type + UTF-8 text: a caught sampling/encode failure

# Host-derived pseudo-types. These are **never put on the wire** — the board
# streams raw MSG_ACCEL only, and the host reconstructs both from it (see
# motion.py). They exist so derived samples can flow through the same
# SensorRecord/naming machinery as decoded ones; they have no SCALES entry
# because they are built directly in SI units, never encoded as fixed point.
MSG_GRAVITY = 0x06       # 3 x float: x, y, z — estimated gravity vector
MSG_LINEAR_ACCEL = 0x07  # 3 x float: x, y, z — acceleration with gravity removed

# Human-readable name + field labels per type, for host-side formatting.
# `battery` carries a trailing u8 (usb_connected) after its two scaled ints;
# `error` carries a source-type byte followed by a text message (no scaling).
MSG_INFO = {
    MSG_ACCEL: ("accel", ("x", "y", "z")),
    MSG_GYRO: ("gyro", ("x", "y", "z")),
    MSG_MAG: ("mag", ("x", "y", "z")),
    MSG_BATTERY: ("battery", ("voltage_v", "percent", "usb_connected")),
    MSG_ERROR: ("error", ("source", "message")),
    MSG_GRAVITY: ("gravity", ("x", "y", "z")),
    MSG_LINEAR_ACCEL: ("linear_accel", ("x", "y", "z")),
}

# Fixed-point scale per scaled int32 field: SI value = raw_int / scale.
# Ranges stay well within int32. Fields not listed here (battery usb flag,
# error message, the host-derived gravity/linear_accel) pass through unscaled.
SCALES = {
    MSG_ACCEL: (1000, 1000, 1000),        # milli-m/s^2         -> 0.001 m/s^2
    MSG_GYRO: (10000, 10000, 10000),      # 1e-4 rad/s
    MSG_MAG: (100, 100, 100),             # 0.01 uT
    MSG_BATTERY: (1000, 100),             # millivolt, 0.01 %
}

# NB: use struct.pack/unpack (functions), not struct.Struct — CircuitPython's
# `struct` provides only the functions, not the Struct class.
_TS_FMT = "<I"  # 4-byte little-endian unsigned timestamp (ms)


def to_raw(msg_type, si_values):
    """Scale SI floats to the int32s put on the wire (device side)."""
    scales = SCALES.get(msg_type, ())
    return [
        int(round(v * scales[i])) if i < len(scales) else int(v)
        for i, v in enumerate(si_values)
    ]


# --- COBS --------------------------------------------------------------------
def cobs_encode(data):
    """Encode ``data`` so the result contains no ``0x00`` bytes."""
    out = bytearray()
    code_index = 0
    out.append(0)  # placeholder for the first code byte
    code = 1
    for byte in data:
        if byte != 0:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_index] = code
                code_index = len(out)
                out.append(0)  # next placeholder
                code = 1
        else:
            out[code_index] = code
            code_index = len(out)
            out.append(0)  # next placeholder
            code = 1
    out[code_index] = code
    return bytes(out)


def cobs_decode(data):
    """Inverse of :func:`cobs_encode`. Raises ValueError on malformed input."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        code = data[i]
        if code == 0 or i + code > n:
            raise ValueError("malformed COBS frame")
        i += 1
        out.extend(data[i:i + code - 1])
        i += code - 1
        if code < 0xFF and i < n:
            out.append(0)
    return bytes(out)


# --- Encode (device side) ----------------------------------------------------
def encode(msg_type, timestamp_ms, ints=(), extra_u8=None):
    """Build one framed TLV record ready to write to the serial port.

    ``ints`` are the already-scaled int32 sensor values (see :func:`to_raw`);
    ``extra_u8`` (used by the battery message for the usb-connected flag)
    appends a single byte.
    """
    value = bytearray()
    value += struct.pack(_TS_FMT, timestamp_ms & 0xFFFFFFFF)
    for n in ints:
        value += struct.pack("<i", n)
    if extra_u8 is not None:
        value.append(extra_u8 & 0xFF)
    payload = bytes((msg_type, len(value))) + bytes(value)
    return cobs_encode(payload) + b"\x00"


def encode_error(timestamp_ms, source_type, message):
    """Build an ERROR frame describing a caught failure.

    ``source_type`` is the message type of the stream that failed (or 0 if
    unknown); ``message`` is free text (truncated to fit the u8 length field).
    """
    text = str(message).encode("utf-8")[:200]
    value = bytearray()
    value += struct.pack(_TS_FMT, timestamp_ms & 0xFFFFFFFF)
    value.append(source_type & 0xFF)
    value += text
    payload = bytes((MSG_ERROR, len(value))) + bytes(value)
    return cobs_encode(payload) + b"\x00"


# --- Decode (host side) ------------------------------------------------------
def parse_payload(payload):
    """Parse a decoded (un-COBS'd) payload into ``(msg_type, timestamp_ms, values)``.

    ``values`` is a tuple of the raw int32 sensor values (still scaled — the
    host applies :data:`SCALES`), with a trailing int for the battery's
    usb-connected flag. For ``MSG_ERROR`` it is ``(source_type_int, message_str)``.
    Raises ValueError on a structurally invalid payload.
    """
    if len(payload) < 2:
        raise ValueError("payload too short")
    msg_type = payload[0]
    length = payload[1]
    value = payload[2:]
    if len(value) != length:
        raise ValueError("length mismatch (declared %d, got %d)" % (length, len(value)))
    if length < 4:
        raise ValueError("value shorter than timestamp")
    (timestamp_ms,) = struct.unpack(_TS_FMT, value[:4])
    body = value[4:]

    if msg_type == MSG_ERROR:
        if len(body) < 1:
            raise ValueError("error frame missing source byte")
        source_type = body[0]
        message = bytes(body[1:]).decode("utf-8", "replace")
        return msg_type, timestamp_ms, (source_type, message)

    has_u8 = msg_type == MSG_BATTERY
    int_bytes = len(body) - (1 if has_u8 else 0)
    if int_bytes < 0 or int_bytes % 4 != 0:
        raise ValueError("value payload not int32-aligned")
    n_ints = int_bytes // 4
    values = list(struct.unpack("<%di" % n_ints, body[:int_bytes])) if n_ints else []
    if has_u8:
        values.append(body[int_bytes])
    return msg_type, timestamp_ms, tuple(values)


class FrameDecoder:
    """Accumulates raw serial bytes and yields parsed records.

    Feed arbitrary chunks to :meth:`feed`; it splits on the ``0x00`` delimiter,
    COBS-decodes and TLV-parses each complete frame, and yields
    ``(msg_type, timestamp_ms, values)`` tuples. Malformed frames are skipped
    (counted in :attr:`errors`) rather than raising, so a glitchy stream
    self-recovers at the next delimiter.
    """

    def __init__(self):
        self._buf = bytearray()
        self.errors = 0

    def feed(self, chunk):
        self._buf += chunk
        while True:
            idx = self._buf.find(0)
            if idx < 0:
                break
            frame = bytes(self._buf[:idx])
            del self._buf[:idx + 1]
            if not frame:
                continue  # empty frame (e.g. leading delimiter) — ignore
            try:
                yield parse_payload(cobs_decode(frame))
            except ValueError:
                self.errors += 1
