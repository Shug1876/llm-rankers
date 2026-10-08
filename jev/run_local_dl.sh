#!/usr/bin/env bash
# Re-rank TREC DL 2019 + 2020 (BM25 top-100) with the open JEV model served by serve_jev9b.sbatch, every method the
# local provider supports, then evaluate. Client side only (no GPU): run it on the login node, it survives logout:
#
#   nohup bash jev/run_local_dl.sh > jev/logs/run_local_dl.log 2>&1 &
#   tail -f jev/logs/run_local_dl.log
#   DATASETS=dl20 JEV_LOCAL_URL=... JEV_LONG_URL=... nohup bash jev/run_local_dl.sh ...   # one dataset per server
#
# Finished runs are skipped, so after a crash or a server restart just run it again. Not run: listwise --mode choice
# (windows of 20/100 exceed the open model's 16 choice options). The 100-in-1 listwise runs (~19k tokens per request)
# go to a second server started with MAX_MODEL_LEN=32768 (JEV_LONG_URL, default jev/logs/jev9b-32k.endpoint):
#   ENDPOINT_FILE=jev/logs/jev9b-32k.endpoint MAX_MODEL_LEN=32768 sbatch --job-name=jev9b-32k jev/serve_jev9b.sbatch
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY=${PYTHON:-/mnt/scratch/users/3148123l/venvs/jev-vllm/bin/python}
export JEV_LOCAL_URL=${JEV_LOCAL_URL:-$(cat jev/logs/jev9b.endpoint)}
export JEV_MODEL_DIR=${JEV_MODEL_DIR:-/mnt/scratch/users/3148123l/models/JEV-9B}
OUT=${OUT:-jev/runs/local9b}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"

LONG_FILE=${LONG_FILE:-jev/logs/jev9b-32k.endpoint}

up() { curl --noproxy '*' -sf -m 10 "$1/v1/models" > /dev/null; }
up "$JEV_LOCAL_URL" || { echo "JEV server not reachable at $JEV_LOCAL_URL (is the serve_jev9b job running?)" >&2; exit 1; }
echo "server $JEV_LOCAL_URL, output $OUT, started $(date)"

long_url() {  # the 32k server; wait up to 20 min for it to start
  for _ in $(seq 120); do
    url=${JEV_LONG_URL:-$(cat "$LONG_FILE" 2>/dev/null || true)}
    if [ -n "$url" ] && up "$url"; then echo "$url"; return 0; fi
    sleep 10
  done
  return 1
}

METHODS=(
  "pointwise.noul            pointwise --method noul"
  "pointwise.score           pointwise --method score"
  "pointwise.cookbook        pointwise --method cookbook"
  "pointwise.cookbook_score  pointwise --method cookbook_score"
  "pointwise.trec            pointwise --method trec"
  "pointwise.umbrela         pointwise --method umbrela"
  "pointwise.grade4          pointwise --method grade4"     # open-model corpus format (retrieval_relevance)
  "pointwise.scenario        pointwise --method scenario"
  "setwise.heapsort.c10      setwise --num_child 10 --k 10"
  "listwise.score.w20s10     listwise --window_size 20 --step_size 10 --mode score"
  "listwise.score.w100       listwise --window_size 100 --step_size 100 --mode score"
  "pairwise.heapsort         pairwise --k 10"
)

for DS in ${DATASETS:-dl19 dl20}; do
  case "$DS" in
    dl19) DATASET=msmarco-passage/trec-dl-2019/judged ;;
    dl20) DATASET=msmarco-passage/trec-dl-2020/judged ;;
  esac
  common="run --provider local --run_path jev/runs/bm25/run.rank_llm.bm25.$DS.top100.txt
          --docs_file jev/runs/bm25/docs.$DS.top100.tsv --ir_dataset_name $DATASET --hits 100 --query_length 32
          --passage_length 128 --num_workers 32 --query_workers 12 --max_rps 0 $EXTRA"
  for m in "${METHODS[@]}"; do
    read -r key args <<< "$m"
    save=$OUT/$DS.$key.txt
    if [ -s "$save.stats.json" ]; then echo "skip $save (done)"; continue; fi
    url=$JEV_LOCAL_URL
    if [[ $key == *w100 ]]; then
      url=$(long_url) || { echo "skip $save: no 32k-context server reachable"; continue; }
    fi
    echo "=== $DS $key  $(date +%T)  $url"
    # tqdm redraws go to stderr; keep only the summary lines in the log
    $PY jev/run_jev.py $common --base_url "$url" --save_path "$save" $args 2> >(grep -v 're-ranking' >&2)
  done
  $PY jev/eval_run.py --dataset $DATASET jev/runs/bm25/run.rank_llm.bm25.$DS.top100.txt "$OUT"/$DS.*.txt
done

[ -n "${DATASETS:-}" ] || JEV_RUNS_DIR=$OUT $PY jev/make_table.py   # full table only when both datasets ran here
echo "finished $(date)"
