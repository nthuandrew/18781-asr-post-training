#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data}"
mkdir -p "$DATA_ROOT"

echo "Downloading open-source ASR datasets into $DATA_ROOT"

echo "This is a scaffold. Replace with concrete dataset download scripts for Librispeech, FLEURS, TED-LIUM, and Numb3rs."
echo "Examples:"
echo "  huggingface-cli download google/fleurs --repo-type dataset --local-dir $DATA_ROOT/fleurs"
echo "  python -c 'from datasets import load_dataset; load_dataset(\"librispeech_asr\", \"all\")'"

exit 0
