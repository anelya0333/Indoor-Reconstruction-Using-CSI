#!/usr/bin/env bash
###############################################################################
# 04_resize_images.sh
#
# Description:
#   Batch-resizes all extracted frame images to a fixed square dimension
#   (default: 256×256 pixels). This step ensures consistent input dimensions
#   for downstream machine learning models or data normalization pipelines.
#
#   The script operates **in-place** within `dataset_out/frames`, replacing
#   each image file after resizing.
#
#   Preferred backend:
#     - Uses ImageMagick's `mogrify` if available (fast, parallel-safe)
#     - Falls back to `ffmpeg` loop (slower but portable)
#
# -----------------------------------------------------------------------------
# Usage:
#   ./04_resize_images.sh [size]
#
#   Examples:
#     ./04_resize_images.sh
#         → Resizes all .jpg images in dataset_out/frames to 256×256 (default)
#
#     ./04_resize_images.sh 128
#         → Resizes to 128×128 pixels
#
# -----------------------------------------------------------------------------
# Requirements:
#   - ImageMagick (`mogrify`) OR ffmpeg must be installed and accessible in PATH
#
#   ImageMagick example install:
#       sudo apt install imagemagick
#
#   ffmpeg example install:
#       sudo apt install ffmpeg
#
# -----------------------------------------------------------------------------
# Input directory:
#   dataset_out/frames/
#       frame_000001.jpg
#       frame_000002.jpg
#       ...
#
# Output (in-place overwrite):
#   dataset_out/frames/
#       frame_000001.jpg  → resized to SIZE×SIZE
#       frame_000002.jpg  → resized to SIZE×SIZE
#
# -----------------------------------------------------------------------------
# Notes:
#   - Uses “!” in mogrify resize syntax to force exact size (ignores aspect ratio)
#   - ffmpeg fallback uses high-quality area-based scaling
#   - Operates safely with `set -euo pipefail` to halt on any failure
#
# Author:
#   (you)
###############################################################################

# --- Safety configuration -----------------------------------------------------
set -euo pipefail
# -e : Exit immediately on command error
# -u : Treat unset variables as errors
# -o pipefail : Propagate errors in pipeline commands

# --- Parameters ---------------------------------------------------------------
FRAMES_DIR="dataset_out/frames"       # Directory containing extracted video frames
SIZE="${1:-256}"                      # Target resize dimension (default = 256)

# --- Execution logic ----------------------------------------------------------
if command -v mogrify >/dev/null 2>&1; then
  ###########################################################################
  # MODE 1: ImageMagick is available → use mogrify for in-place batch resize
  #
  # Flags:
  #   -resize "${SIZE}x${SIZE}!" : Resize to exact WxH pixels, ignoring aspect ratio.
  #   "${FRAMES_DIR}"/*.jpg     : Apply to all JPG images in the frames directory.
  #
  # mogrify edits files in place — it’s much faster than per-file loops.
  ###########################################################################
  mogrify -resize "${SIZE}x${SIZE}!" "${FRAMES_DIR}"/*.jpg
  echo "✅ Resized all images to ${SIZE}x${SIZE} pixels in ${FRAMES_DIR}"

else
  ###########################################################################
  # MODE 2: Fallback → use ffmpeg if ImageMagick is unavailable
  #
  # This approach loops through all .jpg images and rescales them one-by-one.
  # It’s slower but more portable since ffmpeg is widely available.
  #
  # Command details:
  #   -hide_banner : Suppress ffmpeg startup info
  #   -y           : Overwrite output without prompt
  #   -vf "scale=${SIZE}:${SIZE}:flags=area"
  #                   → Resize using area averaging (better for downscaling)
  #
  # To preserve atomic updates, output is written to a temporary file first.
  ###########################################################################
  echo "⚠️  ImageMagick not found; falling back to ffmpeg loop (slower)."
  for f in "${FRAMES_DIR}"/*.jpg; do
    # Generate a temporary filename (same name with '_tmp' suffix)
    tmp="${f%.jpg}_tmp.jpg"

    # Resize the image
    ffmpeg -hide_banner -loglevel error -y -i "$f" \
           -vf "scale=${SIZE}:${SIZE}:flags=area" "$tmp"

    # Replace original image atomically
    mv "$tmp" "$f"
  done
  echo "✅ Resized all images to ${SIZE}x${SIZE} pixels via ffmpeg fallback"
fi

# --- Summary ------------------------------------------------------------------
echo "All frame images are now standardized to ${SIZE}x${SIZE} pixels."
echo "Ready for training, feature extraction, or normalization."
