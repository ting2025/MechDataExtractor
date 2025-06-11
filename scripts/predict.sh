#!/usr/bin/env bash
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <checkpoint.pth> [extra predict.py flags]"
  exit 1
fi

MODEL="$1"
shift
EXTRA_ARGS="$@"        # e.g. --scale 0.5 --mask-threshold 0.5

IN_DIR="test"
OUT_DIR="output/test"

# 1) Activate your conda env (adjust to your install path if needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate imgseg

# 2) Make sure output dir exists
mkdir -p "$OUT_DIR"

# 3) Find all .png/.jpg/.jpeg under IN_DIR, even with spaces
find "$IN_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print0 \
| while IFS= read -r -d '' img; do
    # derive a base name without extension
    base="$(basename "$img")"
    name="${base%.*}"
    out="$OUT_DIR/${name}_OUT.png"

    echo "Predicting:"
    echo "   input:  $img"
    echo "   output: $out"
    python predict.py \
      --model "$MODEL" \
      -i "$img" \
      -o "$out" \
      --bilinear \
      --classes 1 \
      $EXTRA_ARGS
done
