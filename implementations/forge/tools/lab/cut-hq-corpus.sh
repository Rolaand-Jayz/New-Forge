#!/bin/bash
# Re-cut the test tiers from the 2160p lossless references at CRF 10
# (near-transparent, ~4-5x the crf23 bitrate). Frame-exact by construction:
# variant and reference share one cutting chain, killing the rooftop-f120
# misalignment class. Keeps MV export (H.264 inter prediction intact).
# Old crf23 corpus is KEPT for future compression-artifact-removal studies.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
R=research/Temporal-Forge-Player/benchmarks/video_corpus/references
C=research/Temporal-Forge-Player/benchmarks/video_corpus/clips
for scene in tos_daylight tos_debris sintel_rooftop sintel_cave; do
  for tier in 426x240 640x360 854x480 1280x720; do
    W=${tier%x*}; H=${tier#*x}
    out="$C/${scene}_${tier}_hq_crf10.mp4"
    ffmpeg -y -v error -i "$R/${scene}_2160p_lossless.mkv" \
      -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1" \
      -c:v libx264 -preset medium -crf 10 -pix_fmt yuv420p -bf 0 \
      -g 48 -keyint_min 48 -sc_threshold 0 -movflags +faststart -an "$out"
    n=$(ffprobe -v error -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 -count_frames "$out")
    echo "$out frames=$n"
  done
done
# RF/CRF 0 lossless downsample tier (bit-exact decodes; index-aligned)
for scene in tos_daylight tos_debris sintel_rooftop sintel_cave; do
  for tier in 426x240 640x360 854x480 1280x720; do
    W=${tier%x*}; H=${tier#*x}
    out="$C/${scene}_${tier}_lossless_crf0.mp4"
    ffmpeg -y -v error -i "$R/${scene}_2160p_lossless.mkv" \
      -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1" \
      -c:v libx264 -preset medium -crf 0 -pix_fmt yuv420p -bf 0 \
      -g 48 -keyint_min 48 -sc_threshold 0 -movflags +faststart -an "$out"
    n=$(ffprobe -v error -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 -count_frames "$out")
    echo "$out frames=$n"
  done
done
echo CUT-DONE
