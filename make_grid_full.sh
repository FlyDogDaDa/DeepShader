#!/bin/bash

FFMPEG="/home/b11223209/workspace/GanDeepShader/ffmpeg"
SAMPLES_DIR="/home/b11223209/workspace/ProgramDevelopment/DeepShader/runs/transformer-4-subset-32_20260606_001759/samples"
OUTPUT_DIR="/home/b11223209/workspace/ProgramDevelopment/DeepShader/runs/transformer-4-subset-32_20260606_001759"

# Generate file list with absolute paths
> /tmp/samples.txt
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
  for j in 0 1 2 3 4; do
    printf "file '%s/sample_%04d_idx%d.png'\n" "$SAMPLES_DIR" "$i" "$j" >> /tmp/samples.txt
  done
done

echo "Generated $(wc -l < /tmp/samples.txt) entries"

# Build filter: 5 images per sample (hstack), 26 samples stacked (vstack)
filter=""
vnames=()
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
  start=$((i * 5))
  end=$((start + 4))
  if [ -z "$filter" ]; then
    filter="[${start}:${end}]hstack=inputs=5[v${i}]"
  else
    filter="${filter};[${start}:${end}]hstack=inputs=5[v${i}]"
  fi
  vnames+=("v${i}")
done

# Build vstack arguments
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

echo "Running ffmpeg..."
$FFMPEG -hide_banner -loglevel error -f concat -safe 0 -i /tmp/samples.txt \
  -filter_complex "$filter" \
  -map "[out]" "${OUTPUT_DIR}/samples_grid.png"

echo "Done! Output: ${OUTPUT_DIR}/samples_grid.png"
