#!/usr/bin/env python3
"""
00_feitcsi_bin_to_csv.py (fixed)

Parse FeitCSI *binary* capture into a CSV:
  epoch_s, csi_0, csi_1, ..., csi_(R*T*Nsc - 1)

Fixes:
- Enforces a constant feature vector length across the file (skips mismatches).
- Optional strict shape filters: --expect-rx/--expect-tx/--expect-nsc

Usage:
  python 00_feitcsi_bin_to_csv.py \
      --in csi_capture/csi_capture.dat.gz \
      --out dataset_out/csi_raw.csv \
      --layout rx_tx_nsc --avg-by none \
      [--expect-rx 3 --expect-tx 2 --expect-nsc 80]
"""

import argparse, gzip, io, os, struct, sys
from typing import BinaryIO, Tuple, Optional

HEADER_SIZE = 272

def open_auto(path: str) -> BinaryIO:
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")

def read_exact(fp: BinaryIO, n: int) -> Optional[bytes]:
    buf = fp.read(n)
    if not buf or len(buf) < n:
        return None
    return buf

def parse_header(h: bytes) -> dict:
    # little-endian
    size_csi, = struct.unpack_from("<I", h, 0)
    ftm_clock, = struct.unpack_from("<I", h, 8)
    ts_us, = struct.unpack_from("<Q", h, 12)    # 8-byte (uint64) μs timestamp
    n_rx = h[46]
    n_tx = h[47]
    n_sc, = struct.unpack_from("<I", h, 52)
    return {
        "size_csi": size_csi,
        "ftm_clock": ftm_clock,
        "ts_us": ts_us,
        "n_rx": n_rx,
        "n_tx": n_tx,
        "n_sc": n_sc,
    }

def parse_csi_block(raw: bytes, n_rx: int, n_tx: int, n_sc: int):
    """
    raw: bytes for one CSI block. Expected length: 4 * n_rx * n_tx * n_sc
    Each complex sample = (int16 real, int16 imag), little-endian
    Returns amplitudes as a 1D float list in the raw order.
    """
    import numpy as np

    expected = 4 * n_rx * n_tx * n_sc
    if len(raw) != expected:
        raise ValueError(f"CSI byte size mismatch: got {len(raw)} vs expected {expected} "
                         f"(rx={n_rx}, tx={n_tx}, nsc={n_sc})")

    # Interpret as int16 little-endian pairs
    arr = np.frombuffer(raw, dtype="<i2")  # int16 little-endian
    if arr.size != 2 * n_rx * n_tx * n_sc:
        raise ValueError("Unexpected int16 count in CSI data")
    arr = arr.reshape(-1, 2)  # [N, 2] => (real, imag)
    amp = (arr[:,0].astype("float32")**2 + arr[:,1].astype("float32")**2) ** 0.5  # magnitude

    return amp  # shape: [n_rx*n_tx*n_sc]

def reshape_amplitudes(amp, n_rx, n_tx, n_sc, layout: str, avg_by: str):
    """
    amp is 1D length n_rx*n_tx*n_sc in raw order. Canonical reshape: (n_rx, n_tx, n_sc)
    """
    import numpy as np
    x = amp.reshape(n_rx, n_tx, n_sc)

    if avg_by == "tx":
        x = x.mean(axis=1)       # (n_rx, n_sc)
    elif avg_by == "rx":
        x = x.mean(axis=0)       # (n_tx, n_sc)
    elif avg_by != "none":
        raise ValueError("--avg-by must be one of: none, tx, rx")

    if layout == "flat":
        return x.reshape(-1)
    elif layout in ("rtx_nsc", "rx_tx_nsc"):
        x2d = x.reshape(-1, x.shape[-1])  # (*, n_sc)
        return x2d.reshape(-1)
    else:
        raise ValueError("--layout must be one of: flat, rtx_nsc, rx_tx_nsc")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--layout", default="rx_tx_nsc", choices=["flat","rtx_nsc","rx_tx_nsc"])
    ap.add_argument("--avg-by", default="none", choices=["none","tx","rx"])
    ap.add_argument("--start", type=int, default=0, help="skip first N blocks")
    ap.add_argument("--limit", type=int, default=-1, help="max blocks to parse; -1 = all")

    # NEW: optional strict shape filters
    ap.add_argument("--expect-rx", type=int, default=None)
    ap.add_argument("--expect-tx", type=int, default=None)
    ap.add_argument("--expect-nsc", type=int, default=None)
    args = ap.parse_args()

    out = open(args.out, "w", encoding="utf-8")
    out.write("epoch_s")
    wrote_header = False
    expected_vec_len = None

    blocks = 0
    kept = 0
    skipped_shape = 0

    with open_auto(args.inp) as fp:
        while True:
            h = read_exact(fp, HEADER_SIZE)
            if h is None:
                break  # EOF
            hdr = parse_header(h)

            size = hdr["size_csi"]
            n_rx, n_tx, n_sc = hdr["n_rx"], hdr["n_tx"], hdr["n_sc"]
            ts_us = hdr["ts_us"]

            # Optional strict shape filter
            if (args.expect_rx is not None and n_rx != args.expect_rx) or \
               (args.expect_tx is not None and n_tx != args.expect_tx) or \
               (args.expect_nsc is not None and n_sc != args.expect_nsc):
                data = read_exact(fp, size)
                if data is None:
                    break
                blocks += 1
                skipped_shape += 1
                continue

            expected = 4 * n_rx * n_tx * n_sc
            if size != expected:
                # Skip mismatched size blocks
                print(f"[WARN] size_csi={size} != 4*{n_rx}*{n_tx}*{n_sc}={expected}. Skipping block {blocks}.",
                      file=sys.stderr)
                data = read_exact(fp, size)
                if data is None:
                    break
                blocks += 1
                continue

            data = read_exact(fp, size)
            if data is None:
                break

            blocks += 1
            if blocks <= args.start:
                continue

            if args.limit >= 0 and kept >= args.limit:
                break

            try:
                amp = parse_csi_block(data, n_rx, n_tx, n_sc)
                vec = reshape_amplitudes(amp, n_rx, n_tx, n_sc, args.layout, args.avg_by)
            except Exception as e:
                print(f"[WARN] parse error at block {blocks}: {e}", file=sys.stderr)
                continue

            epoch_s = ts_us / 1e6  # μs -> s

            if not wrote_header:
                expected_vec_len = len(vec)
                for i in range(expected_vec_len):
                    out.write(f",csi_{i}")
                out.write("\n")
                wrote_header = True
            else:
                if len(vec) != expected_vec_len:
                    print(
                        f"[WARN] skipping block {blocks}: feature length {len(vec)} "
                        f"!= expected {expected_vec_len} (rx={n_rx}, tx={n_tx}, nsc={n_sc})",
                        file=sys.stderr
                    )
                    skipped_shape += 1
                    continue

            out.write(f"{epoch_s}")
            out.write("," + ",".join(f"{float(v):.6f}" for v in vec) + "\n")
            kept += 1

    out.close()
    print(f"Read blocks: {blocks} | Wrote rows: {kept} | Skipped (shape/size): {skipped_shape} -> {args.out}")

if __name__ == "__main__":
    main()
