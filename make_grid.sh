#!/bin/bash
cd "$(dirname "$0")"

FFMPEG="/home/b11223209/workspace/GanDeepShader/ffmpeg"
SAMPLES_DIR="runs/transformer-4-subset-32_20260606_001759/samples"

# Generate file list
for i in $(seq -w 0 25); do
  for j in 0 1 2 3 4; do
    echo "file '${PWD}/${SAMPLES_DIR}/sample_${i}_idx${j}.png'"
  done
done > /tmp/samples.txt

# Build filter: 5 images per row, 26 rows stacked vertically
# Each block of 5 images per sample
filter=""
vnames=()
idx=0
for i in $(seq -w 0 25); do
  start=$idx
  end=$((idx + 4))
  if [ -z "$filter" ]; then
    filter="[${start}:${end}]hstack=inputs=5[v${i}]"
  else
    filter="${filter};[${start}:${end}]hstack=inputs=5[v${i}]"
  fi
  vnames+=("v${i}")
  idx=$((idx + 5))
done

stack_args=""
for v in "${vnames[@]}"; do
  if [ -z "$stack_args" ]; then
    stack_args="[$v"
  else
    stack_args="${stack_args}[$v"
  fi
done
stack_args="${stack_args}]vstack=inputs=26[out]"

filter="${filter};${stack_args}"

$FFMPEG -hide_banner -loglevel error -f concat -safe 0 -i /tmp/samples.txt \
  -filter_complex "$filter" \
  -map "[out]" samples_grid.png

echo "Done! Output: samples_grid.png"
