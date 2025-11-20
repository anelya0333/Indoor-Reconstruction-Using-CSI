#!/usr/bin/env python3
"""
02_align_and_build_pairs.py (with robust CSV reader)

Creates aligned pairs (frame_path, feature_vector) by:
- reading video_frame_pts.csv (pkt_pts_time = seconds from video start)
- anchoring to wallclock_timestamps.log -> epoch seconds
- reading CSI from dataset_out/csi_raw.csv (epoch_s + amplitudes)
- window-averaging CSI around each frame epoch
- saving manifest.csv and features.npy
"""

import argparse, csv, glob, os, numpy as np
from bisect import bisect_left
import sys

def read_video_pts(csv_path):
    pts = []
    with open(csv_path) as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            try:
                t = float(row[0])     # pkt_pts_time
            except Exception:
                continue
            pts.append(t)
    if not pts:
        raise RuntimeError("video_frame_pts.csv is empty or wrong format")
    return np.array(pts, dtype=np.float64)

def read_wallclock(log_path):
    times = []
    with open(log_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                times.append(float(parts[1]))
            except Exception:
                pass
    if not times:
        raise RuntimeError("wallclock_timestamps.log is empty or wrong format")
    return np.array(times, dtype=np.float64)

def read_csi(csv_path):
    """
    Robust reader: enforces rectangular matrix by ignoring any data row whose
    column count doesn't match the header.
    """
    rows = []
    with open(csv_path) as f:
        header = f.readline().rstrip("\n").split(",")
        if not header or header[0] != "epoch_s":
            raise RuntimeError("First column must be epoch_s")
        expected_cols = len(header)

        dropped = 0
        line_no = 1
        for line in f:
            line_no += 1
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            if len(parts) != expected_cols:
                dropped += 1
                continue
            try:
                rows.append([float(x) for x in parts])
            except Exception:
                dropped += 1

    if not rows:
        raise RuntimeError("No valid CSI rows found (all malformed or empty).")

    mats = np.asarray(rows, dtype=np.float64)
    epoch_s = mats[:, 0].astype(np.float64)
    features = mats[:, 1:].astype(np.float32)

    if dropped > 0:
        print(f"[WARN] read_csi: dropped {dropped} malformed/mismatched rows", file=sys.stderr)

    return epoch_s, features, header[1:]

def list_frames(frames_dir):
    exts = ["*.jpg","*.jpeg","*.png"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frames_dir, e)))
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    return files

def nearest_window_indices(times, center, window_s):
    # binary search for left bound, then collect until > right
    left = bisect_left(times, center - window_s)
    out_idx = []
    t_right = center + window_s
    i = left
    N = len(times)
    while i < N and times[i] <= t_right:
        out_idx.append(i)
        i += 1
    return out_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--video_pts", required=True)
    ap.add_argument("--wallclock", required=True)
    ap.add_argument("--csi_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=1.0, help="frame sampling used when extracting frames")
    ap.add_argument("--window_ms", type=float, default=50.0, help="+/- window around frame time")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frames = list_frames(args.frames_dir)
    v_pts = read_video_pts(args.video_pts)             # relative seconds from video start
    wclock = read_wallclock(args.wallclock)            # epoch seconds (~30Hz logger)
    csi_t, csi_feat, feat_names = read_csi(args.csi_csv)

    # Anchor: assume first video PTS aligns with first wallclock timestamp (start of capture)
    epoch0 = wclock[0]
    frame_epoch_all = epoch0 + v_pts

    # Align the N saved images to N evenly-spaced times across [first,last] of the full video timeline
    first_t, last_t = frame_epoch_all[0], frame_epoch_all[-1]
    target_times = np.linspace(first_t, last_t, num=len(frames))

    # Window averaging in seconds
    win = args.window_ms / 1000.0

    # Build output arrays
    X = np.zeros((len(frames), csi_feat.shape[1]), dtype=np.float32)
    rows = []

    for i, img in enumerate(frames):
        t = float(target_times[i])
        idxs = nearest_window_indices(csi_t, t, win)
        if not idxs:
            j = int(np.argmin(np.abs(csi_t - t)))
            feat = csi_feat[j]
            used = [j]
        else:
            feat = csi_feat[idxs].mean(axis=0)
            used = idxs

        X[i] = feat
        rows.append({
            "index": i,
            "image": os.path.relpath(img),
            "epoch_s": f"{t:.6f}",
            "csi_n": int(len(used)),
        })

    # Save manifest and features
    import json
    with open(os.path.join(args.out_dir, "manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r)+"\n")

    np.save(os.path.join(args.out_dir, "features.npy"), X)
    with open(os.path.join(args.out_dir, "feature_names.txt"), "w") as f:
        for name in feat_names:
            f.write(name+"\n")

    print(f"Wrote: {args.out_dir}/manifest.jsonl")
    print(f"Wrote: {args.out_dir}/features.npy  shape={X.shape}")
    print(f"Wrote: {args.out_dir}/feature_names.txt  ({len(feat_names)} names)")
    print("Hint: next, run 03_normalize_and_split.py, then 04_resize_images.sh")

if __name__ == "__main__":
    main()
