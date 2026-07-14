"""Host-importable helpers for the Feather Sense.

Only the host-side modules (``stream``) are meant to be imported as a package;
``code.py``/``sensors.py`` run on the board and ``feather_protocol.py`` is shared
by both sides (imported as a top-level module via ``stream``/``read_stream``).
"""
