#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# receiver_record.sh
# ---------------------------------------------------------------------------
# Purpose:
#   Concurrently record:
#     1) Wi‑Fi Channel State Information (CSI) via FeitCSI (requires sudo)
#     2) Synchronized room video via ffmpeg (with live wall‑clock overlay)
#
#   While recording, a separate lightweight loop logs high‑frequency wall‑clock
#   timestamps to aid post‑hoc alignment between CSI samples, video frames, and
#   real time. On clean termination (trap INT/TERM), the script:
#     - Extracts per‑frame PTS/DTS timestamps using ffprobe → CSV
#     - Gzips the raw CSI file
#     - Waits for child processes to finish and reports artifact locations
#
# Usage:
#   sudo ./receiver_record.sh          # runs until Ctrl‑C
#   RECORD_DURATION=60 sudo ./receiver_record.sh   # fixed 60‑second capture
#
# Safety & robustness features:
#   - strict shell options (set -euo pipefail)
#   - preflight checks for binaries/devices
#   - best‑effort NTP sync
#   - PID tracking + layered termination (INT → TERM → KILL)
#   - trap‑based cleanup that always runs on exit paths
# ---------------------------------------------------------------------------

# Exit on first error (-e), undefined variable (-u), and failed pipeline (-o pipefail).
set -euo pipefail
# Constrain IFS to newline+tab to avoid word‑splitting surprises.
IFS=$'\n\t'

##### CONFIG #####
# Directory where all outputs for this run will be saved.
OUTPUT_DIR="csi_capture"
# Base filenames (some are later derived/augmented).
CSI_FILENAME="csi_capture.dat"   # compressed to .gz during cleanup
VIDEO_FILENAME="room_video.mp4"
TS_LOG="wallclock_timestamps.log"     # lines: <seq> <epoch.seconds_millis>
FRAME_TS_CSV="video_frame_pts.csv"    # ffprobe export of frame PTS/DTS

# FeitCSI binary and default radio parameters.
FEITCSI="/usr/local/bin/feitcsi"      # path to feitcsi binary
FEITCSI_ARGS=(--mode measure           # receive/measure mode
              --frequency 5745         # center freq (MHz) e.g., ch 149 (5.745 GHz)
              --channel-width 80       # 80 MHz channel
              --format VHT             # 802.11ac (Very High Throughput)
              --output-file)           # final arg is the output file path

# Video capture configuration (V4L2 source via ffmpeg).
CAMERA_DEVICE="/dev/video0"
VIDEO_RESOLUTION="1280x720"
FPS="30"
FFMPEG="/usr/bin/ffmpeg"
# ffprobe is optional; used only during cleanup to derive per‑frame timestamps.
FFPROBE="$(command -v ffprobe || true)"
# Prefer system 'date'; fall back to absolute path if not in PATH.
DATE_CMD=$(command -v date || echo "/bin/date")
# ntpdate is optional; if present we attempt a one‑shot time sync.
NTPDATE_CMD=$(command -v ntpdate || true)

# Optional fixed duration in seconds. When empty, record until Ctrl‑C.
RECORD_DURATION="120"
#################################################

# Ensure output directory exists and switch into it so we can refer to outputs
# by simple filenames. Using absolute $PWD below still records full paths.
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Lightweight timestamped logger: log "message" → "YYYY‑MM‑DD HH:MM:SS message"
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# ----------------------------- Preflight checks ----------------------------
log "Starting preflight checks..."
# Verify feitcsi binary is executable.
[ -x "$FEITCSI" ] || { log "ERROR: feitcsi not executable at $FEITCSI"; exit 2; }
# Verify ffmpeg exists (for video capture).
[ -x "$FFMPEG" ]  || { log "ERROR: ffmpeg not found at $FFMPEG"; exit 2; }
# Verify the camera device node exists and is a character device.
[ -c "$CAMERA_DEVICE" ] || { log "ERROR: camera device $CAMERA_DEVICE not found"; exit 2; }
# Check if our date supports millisecond precision (%3N). Not fatal if not.
if ! $DATE_CMD +%s.%3N >/dev/null 2>&1; then
  log "Warning: your date may not support +%s.%3N; alignment will be lower‑precision."
fi

# Best‑effort clock sync (helps align video overlay w/ logged timestamps)
if [ -n "${NTPDATE_CMD:-}" ] && command -v ntpdate >/dev/null 2>&1; then
  log "Syncing clock with pool.ntp.org (best effort)..."
  sudo ntpdate -u pool.ntp.org || log "ntpdate failed; continuing with local clock"
else
  log "ntpdate not available; ensure clocks are synced if you need sub‑second accuracy."
fi

# Resolve absolute output paths for clarity in logs and in cleanup.
CSI_FILE_PATH="$PWD/$CSI_FILENAME"
VIDEO_FILE_PATH="$PWD/$VIDEO_FILENAME"
TS_LOG_PATH="$PWD/$TS_LOG"
FRAME_TS_CSV_PATH="$PWD/$FRAME_TS_CSV"

log "Output directory: $PWD"
log "CSI output: $CSI_FILE_PATH"
log "Video output: $VIDEO_FILE_PATH"
log "Timestamp log: $TS_LOG_PATH"

# ------------------------------ Cleanup/Trap -------------------------------
# We'll collect child PIDs here so cleanup can stop them in order.
PIDS=()
cleanup() {
  log "Caught termination signal — cleaning up..."

  # Stop feitcsi first so capture quiets down
  if [[ -n "${FEITCSI_PID:-}" ]] && kill -0 "$FEITCSI_PID" 2>/dev/null; then
    log "Stopping FeitCSI (PID $FEITCSI_PID)"
    kill -INT "$FEITCSI_PID" 2>/dev/null || true
    sleep 0.5
    kill -TERM "$FEITCSI_PID" 2>/dev/null || true
  fi

  # Ask ffmpeg to finish cleanly (write moov atom)
  if [[ -n "${FFMPEG_PID:-}" ]] && kill -0 "$FFMPEG_PID" 2>/dev/null; then
    log "Asking ffmpeg (PID $FFMPEG_PID) to finalize..."
    kill -INT "$FFMPEG_PID" 2>/dev/null || true
    # Wait up to ~10s before escalating; MP4 finalization can take a moment
    for i in {1..100}; do
      if ! kill -0 "$FFMPEG_PID" 2>/dev/null; then break; fi
      sleep 0.1
    done
    if kill -0 "$FFMPEG_PID" 2>/dev/null; then
      log "ffmpeg still running; sending TERM and waiting more..."
      kill -TERM "$FFMPEG_PID" 2>/dev/null || true
      for i in {1..50}; do
        if ! kill -0 "$FFMPEG_PID" 2>/dev/null; then break; fi
        sleep 0.1
      done
    fi
    if kill -0 "$FFMPEG_PID" 2>/dev/null; then
      log "ffmpeg unresponsive; sending KILL (data may be lost)"
      kill -KILL "$FFMPEG_PID" 2>/dev/null || true
    fi
  fi

  # Wait to ensure files are flushed
  [[ -n "${FFMPEG_PID:-}" ]] && wait "$FFMPEG_PID" 2>/dev/null || true
  [[ -n "${FEITCSI_PID:-}" ]] && wait "$FEITCSI_PID" 2>/dev/null || true

  # --- Derived artifacts ---
  # Extract per‑frame timestamps using ffprobe; useful to align with TS_LOG.
  if [ -f "$VIDEO_FILE_PATH" ]; then
    log "Extracting per-frame PTS timestamps with ffprobe..."
    if [ -n "$FFPROBE" ] && command -v ffprobe >/dev/null 2>&1; then
      ffprobe -hide_banner -select_streams v:0 -show_frames -print_format csv=p=0 \
        -show_entries frame=pkt_pts_time,pkt_dts_time -i "$VIDEO_FILE_PATH" \
        > "$FRAME_TS_CSV_PATH" 2>/dev/null || log "ffprobe extraction failed."
      log "Frame timestamps -> $FRAME_TS_CSV_PATH"
    else
      log "ffprobe not found; skipping frame timestamp extraction."
    fi
  else
    log "Video file not found; skipping ffprobe."
  fi

  # Compress CSI capture to save space; .dat → .dat.gz
  if [ -f "$CSI_FILE_PATH" ]; then
    log "Compressing CSI file..."
    gzip -f "$CSI_FILE_PATH" || log "gzip failed"
    log "CSI compressed -> ${CSI_FILE_PATH}.gz"
  fi

  log "Cleanup done. Outputs in $PWD"
  exit 0
}
# Ensure cleanup runs on Ctrl‑C (INT) or termination (TERM).
trap cleanup INT TERM

# --------------------- Wall‑clock timestamp logger (~30 Hz) -----------------
# Writes monotonically increasing sequence numbers and wall‑clock times.
# This runs independently of ffmpeg/feitcsi to provide an external timing rail.
log "Starting wall-clock timestamp logger (~30 Hz) -> $TS_LOG_PATH"
seq=0
(
  while true; do
    # Print: "<seq> <epoch.sss>"; keep printf fast and simple.
    printf "%s %s\n" "$seq" "$( $DATE_CMD +%s.%3N )"
    seq=$((seq + 1))
    sleep 0.033   # ~30 Hz; adjust cautiously to avoid drift
  done
) >>"$TS_LOG_PATH" 2>&1 &
TS_LOGGER_PID=$!
PIDS+=("$TS_LOGGER_PID")
log "Timestamp logger PID $TS_LOGGER_PID"

# ------------------------------ Start FeitCSI -------------------------------
# The feitcsi process writes raw CSI to $CSI_FILE_PATH. Output is suppressed
# to keep the console clean (remove redirections for verbose debugging).
log "Starting FeitCSI..."
sudo "$FEITCSI" "${FEITCSI_ARGS[@]}" "$CSI_FILE_PATH" >/dev/null 2>&1 &
FEITCSI_PID=$!
PIDS+=("$FEITCSI_PID")
log "FeitCSI PID $FEITCSI_PID"

# Small delay to let feitcsi initialize before starting video.
sleep 0.5

# -------------------------- Start ffmpeg (video) ----------------------------
# Try to draw a live wall‑clock overlay using a known TTF font. This helps
# visually verify time alignment with the external TS_LOG.
FONTFILE="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if [ ! -f "$FONTFILE" ]; then
  FONTFILE="/usr/share/fonts/truetype/freefont/FreeSans.ttf"
fi
# Build the drawtext filter only if we found a font; otherwise, skip overlay.
if [ -f "$FONTFILE" ]; then
  # Note the heavy escaping required so ffmpeg sees the intended format string
  # with %Y-%m-%d %H:%M:%S.%3f expanded at render time.
  DRAW_TEXT_FILTER="drawtext=fontfile=${FONTFILE}:text='%{localtime\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S.%3f}':x=10:y=10:fontsize=24:box=1:boxcolor=0x00000099:fontcolor=white"
else
  log "Font not found; running without overlay."
  DRAW_TEXT_FILTER=""
fi

log "Starting ffmpeg -> $VIDEO_FILE_PATH"
if [ -n "$DRAW_TEXT_FILTER" ]; then
  "$FFMPEG" -y -f v4l2 -framerate "$FPS" -video_size "$VIDEO_RESOLUTION" -i "$CAMERA_DEVICE" \
    -vf "$DRAW_TEXT_FILTER" -c:v libx264 -preset fast -crf 23 "$VIDEO_FILE_PATH" >/dev/null 2>&1 &
else
  "$FFMPEG" -y -f v4l2 -framerate "$FPS" -video_size "$VIDEO_RESOLUTION" -i "$CAMERA_DEVICE" \
    -c:v libx264 -preset fast -crf 23 "$VIDEO_FILE_PATH" >/dev/null 2>&1 &
fi
FFMPEG_PID=$!
PIDS+=("$FFMPEG_PID")
log "ffmpeg PID $FFMPEG_PID"

# -------------------------- Run duration & wait -----------------------------
if [ -n "${RECORD_DURATION}" ]; then
  # Timed run: sleep for the requested duration then trigger cleanup.
  log "Recording for fixed duration: ${RECORD_DURATION}s..."
  sleep "$RECORD_DURATION"
  log "Duration elapsed; cleaning up..."
  cleanup
else
  # Indefinite run: sleep in a long loop until a signal interrupts us (Ctrl‑C).
  log "Recording until Ctrl-C..."
  while true; do sleep 3600; done
fi
