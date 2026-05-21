#!/usr/bin/env python3
"""
parse_amdgpu_trace.py — turn /tmp/amdgpu_mmio_trace.txt into a binary
write-table that the η.3h smoke replays.

Strategy:
  - Walk the trace line-by-line.
  - Stop at the LOAD_KDB cmd write (C2PMSG_35 := 0x80000 at BAR5 offset
    0x5818c). That marks the end of the "pre-PSP setup" we want to replay.
  - For each W (write) event up to that point:
      * If the address is the MMIO_INDEX register (BAR5 0x0000 / phys
        0xf6b00000), look ahead one step.
        - Followed by a WRITE at MMIO_DATA (0xf6b00004) → this is an
          SMN WRITE; emit both (INDEX, then DATA).
        - Followed by anything else → this is an SMN READ index setup;
          skip. (The read itself does not change device state.)
      * Else: emit the write.
  - Also emit ALL writes (state-changing) including MMHUB, IH, NBIO,
    HDP, GFX, etc.

Output:
  /tmp/amdgpu_replay.bin
    binary layout:
      uint32 N                — count of entries
      N × {uint32 offset, uint32 value}
    offset is BAR5 byte offset (subtract 0xf6b00000 from phys).

Limits:
  Only width-4 writes. (Trace has W 4 ...; no 1/2/8 byte writes were
  observed in our capture.)
"""

import struct
import sys
from pathlib import Path

TRACE_PATH = Path("/tmp/amdgpu_mmio_trace.txt")
OUT_PATH   = Path("/tmp/amdgpu_replay.bin")
BAR5_BASE  = 0xf6b00000
MMIO_INDEX = BAR5_BASE + 0x0000   # SMN INDEX register
MMIO_DATA  = BAR5_BASE + 0x0004   # SMN DATA register

# Stop at the first write to C2PMSG_35 with cmd value 0x80000 (LOAD_KDB).
# All writes BEFORE that line are "pre-PSP setup" — that's what we want
# to replay.
C2PMSG_35  = BAR5_BASE + 0x5818c
LOAD_KDB_CMD = 0x80000


def parse():
    if not TRACE_PATH.exists():
        print(f"trace file missing: {TRACE_PATH}", file=sys.stderr)
        sys.exit(1)

    # Parse all events first so we can do single-step lookahead easily.
    events = []
    for raw in TRACE_PATH.read_text().splitlines():
        if not raw or raw.startswith('#'):
            continue
        parts = raw.split()
        if len(parts) < 6:
            continue
        kind = parts[0]
        if kind not in ('W', 'R'):
            continue
        try:
            width = int(parts[1])
            if width != 4:
                continue
            addr  = int(parts[4], 16)
            value = int(parts[5], 16)
        except ValueError:
            continue
        events.append((kind, addr, value))

    print(f"total parsed events: {len(events)}", file=sys.stderr)

    writes_out = []   # list of (offset_in_bar5, value)
    stopped_at = None

    i = 0
    while i < len(events):
        kind, addr, value = events[i]
        if kind != 'W':
            i += 1
            continue

        # Stop at LOAD_KDB cmd.
        if addr == C2PMSG_35 and value == LOAD_KDB_CMD:
            stopped_at = i
            break

        # SMN INDEX register handling.
        if addr == MMIO_INDEX:
            # Look ahead: if next event is a WRITE to MMIO_DATA, this is
            # an SMN WRITE — emit both. Otherwise skip (SMN read setup).
            if i + 1 < len(events):
                n_kind, n_addr, n_val = events[i + 1]
                if n_kind == 'W' and n_addr == MMIO_DATA:
                    # SMN write — emit INDEX then DATA in order.
                    writes_out.append((addr - BAR5_BASE, value))
                    writes_out.append((n_addr - BAR5_BASE, n_val))
                    i += 2
                    continue
            # SMN read setup — skip (no state change).
            i += 1
            continue

        # MMIO_DATA written without preceding INDEX-write? Treat as plain
        # write (shouldn't happen often; just emit).
        if addr == MMIO_DATA:
            writes_out.append((addr - BAR5_BASE, value))
            i += 1
            continue

        # Plain direct-BAR5 write.
        offset = addr - BAR5_BASE
        if offset < 0 or offset >= 0x100000:
            # Off-BAR5 write (BAR0 / BAR2). Skip — we'll handle those
            # separately if needed.
            i += 1
            continue
        writes_out.append((offset, value))
        i += 1

    if stopped_at is None:
        print("WARNING: LOAD_KDB cmd not found in trace. Replaying everything.",
              file=sys.stderr)

    # Emit binary.
    n = len(writes_out)
    with OUT_PATH.open('wb') as f:
        f.write(struct.pack('<I', n))
        for off, val in writes_out:
            f.write(struct.pack('<II', off, val))

    print(f"wrote {n} write entries to {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)",
          file=sys.stderr)

    # Quick stats by BAR5 region for diagnostics.
    region_counts = {}
    for off, _ in writes_out:
        if off < 0x1000:    r = "0x00000-0x00fff NBIO/RCC"
        elif off < 0x4000:  r = "0x01000-0x03fff misc"
        elif off < 0x40000: r = "0x04000-0x3ffff varied"
        elif off < 0x58000: r = "0x40000-0x57fff GC"
        elif off < 0x59000: r = "0x58000-0x58fff MP0/MP1 PSP/SMU"
        elif off < 0x68000: r = "0x59000-0x67fff extra"
        elif off < 0x70000: r = "0x68000-0x6ffff MMHUB"
        else:               r = "0x70000+ high"
        region_counts[r] = region_counts.get(r, 0) + 1

    print("\nwrites by region:", file=sys.stderr)
    for r, c in sorted(region_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c:5d}  {r}", file=sys.stderr)


if __name__ == "__main__":
    parse()
