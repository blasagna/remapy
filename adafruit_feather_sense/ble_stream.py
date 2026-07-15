"""Host-side BLE reader for the Feather Sense (Nordic UART Service, via bleak).

`FeatherSenseBLEStream` mirrors :class:`adafruit_feather_sense.stream.FeatherSenseStream`
exactly (``poll()`` / ``errors`` / ``close()`` / context manager / ``.port`` /
``open_if_available``) so the rerun/recording apps consume BLE with the same code
path they use for USB serial — only the pipe differs.

bleak is asyncio-based while the app loops are synchronous, so the scan → connect →
notify runs on a private asyncio event loop in a background thread. The UART TX
notification callback feeds bytes into a shared :class:`FrameRecordDecoder` and
pushes decoded records onto a thread-safe queue that ``poll()`` drains. The wire
protocol is identical to serial, so nothing below the transport changes.

Nordic UART Service UUIDs: service ``6E400001-…``, TX (device→host notify)
``6E400003-…``, RX (host→device write) ``6E400002-…``.
"""

import asyncio
import os
import queue
import sys
import threading
import time

from bleak import BleakClient, BleakScanner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream import FrameRecordDecoder  # noqa: E402  (sibling module; see stream.py path note)

NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device -> host (notify)
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # host -> device (write)

DEFAULT_NAME = "FeatherSense"


class FeatherSenseBLEStream:
    """Pump a Feather Sense BLE (Nordic UART) stream from a synchronous loop.

    Scans for the device (by ``address`` if given, else advertised ``name``),
    connects, and subscribes to UART notifications on a background asyncio
    thread. Prefer :meth:`open_if_available` when the device may be absent.
    """

    def __init__(self, address=None, name=DEFAULT_NAME, scan_timeout=6.0):
        self._address = address
        self._name = name
        self._scan_timeout = scan_timeout
        self._decoder = FrameRecordDecoder()
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._failed = threading.Event()
        # Uniform with the serial stream's `.port`; refined to the real address.
        self.port = address or f"{name} (BLE)"
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()

    # --- background asyncio thread -------------------------------------------
    def _run_thread(self):
        try:
            asyncio.run(self._run())
        except Exception:  # noqa: BLE001 - surface as "failed", never crash the app
            self._failed.set()

    async def _run(self):
        if self._address:
            device = await BleakScanner.find_device_by_address(
                self._address, timeout=self._scan_timeout
            )
        else:
            device = await BleakScanner.find_device_by_name(
                self._name, timeout=self._scan_timeout
            )
        if device is None:
            self._failed.set()
            return
        self.port = getattr(device, "address", self.port)
        async with BleakClient(device) as client:
            await client.start_notify(NUS_TX, self._on_notify)
            self._connected.set()
            while not self._stop.is_set() and client.is_connected:
                await asyncio.sleep(0.1)
            try:
                await client.stop_notify(NUS_TX)
            except Exception:  # pragma: no cover - best effort on teardown
                pass

    def _on_notify(self, _char, data):
        # Runs on the asyncio thread; queue is thread-safe.
        for record in self._decoder.feed(bytes(data)):
            self._queue.put(record)

    # --- FeatherSenseStream-compatible surface --------------------------------
    def poll(self):
        """Return the SensorRecords received since the last poll (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    @property
    def connected(self):
        return self._connected.is_set()

    @property
    def errors(self):
        return self._decoder.errors

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @classmethod
    def open_if_available(cls, address=None, name=DEFAULT_NAME, scan_timeout=6.0, probe_timeout=12.0):
        """Return a live stream once the board is found and streaming, else None.

        Starts scanning/connecting immediately and waits up to ``probe_timeout``
        seconds for the first decoded record (covers scan + connect + first
        notification). Records seen while probing are preserved for the first
        :meth:`poll`. Returns ``None`` if the device isn't found or stays silent.
        """
        stream = cls(address=address, name=name, scan_timeout=scan_timeout)
        deadline = time.monotonic() + probe_timeout
        while time.monotonic() < deadline:
            if stream._failed.is_set():
                stream.close()
                return None
            records = stream.poll()
            if records:
                for record in records:  # requeue so the caller's first poll sees them
                    stream._queue.put(record)
                return stream
            time.sleep(0.05)
        stream.close()
        return None
