"""Host-importable helpers for the Feather Sense.

Only the host-side modules (``stream``, ``ble_stream``) are meant to be imported
as a package. The board-side files (``board/serial/code.py``, ``board/ble/code.py``,
``sensors.py``, ``telemetry.py``) run on the device; ``feather_protocol.py`` is
shared by both sides (imported as a top-level module via ``stream``/``read_stream``).

:func:`open_feather` is the transport-agnostic entry point used by the apps.
"""

TRANSPORTS = ("serial", "ble")


def open_feather(transport="serial", *, port=None, address=None, **kwargs):
    """Open a Feather Sense stream over the chosen transport, or return ``None``.

    Returns an object with the shared stream interface (``poll()`` / ``errors`` /
    ``close()`` / ``.port``), or ``None`` if the device isn't detected. Imports the
    transport backend lazily so bleak/pyserial load only when actually used.

    - ``transport="serial"`` → :meth:`stream.FeatherSenseStream.open_if_available`
      (``port`` = serial device, default auto-detect).
    - ``transport="ble"`` → :meth:`ble_stream.FeatherSenseBLEStream.open_if_available`
      (``address`` = BLE address, default scan for the ``FeatherSense`` name).
    """
    if transport == "ble":
        from .ble_stream import FeatherSenseBLEStream
        return FeatherSenseBLEStream.open_if_available(address, **kwargs)
    if transport == "serial":
        from .stream import FeatherSenseStream
        return FeatherSenseStream.open_if_available(port, **kwargs)
    raise ValueError(f"unknown transport {transport!r}; expected one of {TRANSPORTS}")
