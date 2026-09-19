#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_tag="${1:?Usage: bash experiments/inferrelay_two_node/run_t1.sh UNIQUE_RUN_TAG}"
[[ "$run_tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
experiment_python=/home/leocao/miniconda3/envs/llmexp/bin/python
rsync -a --exclude __pycache__ --exclude results experiments/inferrelay_two_node/ inferrelay-b:/home/leocao/InferRelay/FlexLLMGen/experiments/inferrelay_two_node/
experiment_outputs=()
for model in opt-125m opt-1.3b; do
  model_tag="${model//./}"
  for direction in A-B B-A; do
    destination="experiments/inferrelay_two_node/results/${run_tag}-${model_tag}-${direction}"
    "$experiment_python" -u -m experiments.inferrelay_two_node.launch --model "$model" --direction "$direction" --out "$destination"
    experiment_outputs+=("$destination")
  done
done
"$experiment_python" -m experiments.inferrelay_two_node.report --runs "${experiment_outputs[@]}" --out "experiments/inferrelay_two_node/results/${run_tag}-summary"
