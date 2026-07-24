---
name: feather-deploy
description: Deploy/flash the Adafruit Feather Sense board — copy the shared modules plus the chosen serial or BLE code.py to CIRCUITPY, install the circup drivers, and eject cleanly. Use when flashing, reflashing, or switching the board between USB-serial and BLE transports.
---

# Deploy the Feather Sense board

CircuitPython app for the **Adafruit Feather Bluefruit Sense (nRF52840)**. One `code.py` runs at a
time; swap transports by re-copying the other build. Full protocol spec, `circup freeze` output, and
troubleshooting are in `adafruit_feather_sense/README.md` (§ Deploy) — read it if anything below is
ambiguous.

Run everything from the repo root. The board mounts at `/media/bob/CIRCUITPY` (confirm with
`mount | grep -i circuitpy`).

## 1. Install sensor drivers (once)

```bash
pixi run circup install adafruit_lsm6ds adafruit_lis3mdl neopixel
```

For the BLE build, also:

```bash
pixi run circup install adafruit_ble
```

## 2. Copy the shared modules + the chosen transport's code.py

The four shared board modules are `feather_protocol.py`, `sensors.py`, `telemetry.py`,
`status_led.py`. Pick exactly one entry point — it lands on the drive as `code.py`.

USB-serial build (IMU 100 Hz):

```bash
cd adafruit_feather_sense
cp feather_protocol.py sensors.py telemetry.py status_led.py board/serial/code.py /media/bob/CIRCUITPY/ && sync
```

BLE build (Nordic UART peripheral `FeatherSense`, IMU 50 Hz):

```bash
cd adafruit_feather_sense
cp feather_protocol.py sensors.py telemetry.py status_led.py board/ble/code.py /media/bob/CIRCUITPY/ && sync
```

Do **not** copy the host-only files (`stream.py`, `ble_stream.py`, `motion.py`, `read_*.py`,
`__init__.py`) to the board.

## 3. Eject cleanly

CIRCUITPY corrupts easily — never yank the board without unmounting. Find the device node first
(it is **not** always `sdb1` — it depends on what else is plugged in), then unmount it:

```bash
dev=$(mount | grep -i circuitpy | awk '{print $1}')   # e.g. /dev/sdb1
udisksctl unmount -b "$dev"
```

## Notes

- Switching transports = re-copy the other build's `code.py` (step 2); the shared modules are the
  same. A `circup uninstall`/reflash wipes `lib/`, so re-run the `circup install` from step 1 after.
- Once streaming, read it from the host with `pixi run rerun --feather` / `--feather-transport ble`,
  or the standalone `read_stream.py` / `read_ble.py` CLIs. See `adafruit_feather_sense/CLAUDE.md`.
