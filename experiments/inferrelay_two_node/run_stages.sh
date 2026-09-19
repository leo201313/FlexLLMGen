#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
tag="${1:?Usage: run_stages.sh UNIQUE_TAG [all|t15|t2|t3]}"
stage="${2:-all}"
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$stage" =~ ^(all|t15|t2|t3)$ ]] || exit 2
experiment_python=/home/leocao/miniconda3/envs/llmexp/bin/python
base=experiments/inferrelay_two_node
rsync -a --exclude results --exclude __pycache__ "$base/" inferrelay-b:/home/leocao/InferRelay/FlexLLMGen/experiments/inferrelay_two_node/
run_worker() {
  "$experiment_python" -u -m experiments.inferrelay_two_node.launch "$@"
}
if [[ "$stage" == all || "$stage" == t15 ]]; then
  outputs=()
  for direction in A-B B-A; do
    out="$base/results/$tag-t15-$direction"
    run_worker --worker interference --model opt-1.3b --direction "$direction" --repeats 10 --out "$out"
    outputs+=("$out")
  done
  "$experiment_python" -m experiments.inferrelay_two_node.report_interference --runs "${outputs[@]}" --out "$base/results/$tag-t15-summary"
fi
if [[ "$stage" == all || "$stage" == t2 ]]; then
  outputs=()
  for model in opt-125m opt-1.3b; do
    model_tag="${model//./}"
    out="$base/results/$tag-t2base-$model_tag-A-B"
    run_worker --worker model_worker --model "$model" --config "$base/configs/t2.json" --out "$out"
    outputs+=("$out")
    for direction in A-B B-A; do
      out="$base/results/$tag-t2-$model_tag-$direction"
      run_worker --worker model_worker --model "$model" --direction "$direction" --config "$base/configs/t2-prefetch.json" --out "$out"
      outputs+=("$out")
    done
  done
  "$experiment_python" -m experiments.inferrelay_two_node.verify_partitions --out "$base/results/$tag-t2-unequal"
  "$experiment_python" -m experiments.inferrelay_two_node.report_model --runs "${outputs[@]}" --out "$base/results/$tag-t2-summary"
fi
if [[ "$stage" == all || "$stage" == t3 ]]; then
  # Require this tag's correctness gates and identical numerical executor source.
  "$experiment_python" - "$tag" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path('experiments/inferrelay_two_node');tag=sys.argv[1]
for model in ['opt-125m','opt-1.3b']:
 for direction in ['A-B','B-A']:
  model_tag=model.replace('.', '')
  run=root/'results'/f'{tag}-t2-{model_tag}-{direction}'
  assert json.loads((run/'launcher_completion.json').read_text())['success']
  for node in ['A','B']:
   assert json.loads((run/node/'completion.json').read_text())['correct']
   hashes=json.loads((run/node/'source_sha256.json').read_text())
   for name in ['executor.py','model_worker.py','wire.py']:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==hashes[name],name
print('T2 correctness gates and source versions verified')
PY
  outputs=()
  for model in opt-125m opt-1.3b; do
    model_tag="${model//./}"
    for direction in A-B B-A; do
      out="$base/results/$tag-t3-$model_tag-$direction"
      run_worker --worker model_worker --model "$model" --direction "$direction" --config "$base/configs/t3.json" --out "$out"
      outputs+=("$out")
    done
  done
  "$experiment_python" -m experiments.inferrelay_two_node.report_model --runs "${outputs[@]}" --out "$base/results/$tag-t3-summary"
fi
