# First-stage retriever benchmark

Top-100 first-stage runs, evaluated as they are produced, for four retrievers over five benchmarks. Everything is
built on [PyTerrier](https://github.com/terrier-org/pyterrier):

| Retriever | English sets (BEIR, MS MARCO, BRIGHT, CoIR, MIRACL-en) | MIRACL (other 17 languages) | Backend |
|---|---|---|---|
| `bm25` | BM25 k1=0.9 b=0.4, porter2, Terrier stopwords | BM25 k1=0.9 b=0.4 over Unicode tokens (see below) | [PISA](https://github.com/terrierteam/pyterrier_pisa) |
| `splade` | [`naver/splade-v3`](https://huggingface.co/naver/splade-v3) (gated: accept the licence) | [`opensearch-neural-sparse-encoding-multilingual-v1`](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1) | PISA, impact-scored |
| `e5` | [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) | [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) | [pyterrier-dr](https://github.com/terrierteam/pyterrier_dr) FlexIndex, exact search |
| `colbert` | [`lightonai/colbertv2.0`](https://huggingface.co/lightonai/colbertv2.0) | [`lightonai/mLateOn`](https://huggingface.co/lightonai/mLateOn) | [pyterrier-pylate](https://github.com/lightonai/pyterrier-pylate) PLAID (fast-plaid) |

| Benchmark | Datasets (`--dataset` names) | Source | Measures |
|---|---|---|---|
| BEIR | 13 public sets + 12 CQADupStack forums (`beir/...`) | ir_datasets `beir/*` | nDCG@10, R@100 |
| MS MARCO | `msmarco-dev` (dev/small, 6,980 queries), `msmarco-dl19` (43), `msmarco-dl20` (54); one shared index | ir_datasets `msmarco-passage/dev/small`, `msmarco-passage/trec-dl-{2019,2020}/judged` | dev: RR@10, nDCG@10, R@100; DL: nDCG@10, RR(rel=2)@10, AP(rel=2)@100, R(rel=2)@100 |
| BRIGHT | 12 domains (`bright/...`) | ir_datasets `bright/*` + `excluded_ids` from `xlangai/BRIGHT` | nDCG@10, R@100 |
| MIRACL | 18 languages, dev (`miracl/<lang>`) | ir_datasets `miracl/<lang>/dev` | nDCG@10, R@100 |
| CoIR | 10 tasks; CodeSearchNet and CodeSearchNet-CCR per language (`coir/...`) | HF `CoIR-Retrieval/*` (test qrels) | nDCG@10, R@100 |

Run `python retrievers/run.py --list` for every name. Licensed BEIR sets (BioASQ, Signal-1M, TREC-NEWS, Robust04)
are not included.

## Setup

```bash
bash retrievers/setup.sh
```

This creates the uv venv `/mnt/scratch/users/3148123l/venvs/retrievers` (Python 3.11; override with `VENV=...`),
installs [requirements.txt](requirements.txt), then `pyterrier-pylate` from GitHub in a separate step, checks every
backend imports, and creates `retrievers/logs/`. `pyterrier-pylate` is not on PyPI; if GitHub is unreachable that
step only prints a warning and the venv still runs BM25, SPLADE and E5 (only `--retriever colbert` needs it).
Run it on the login node (it has internet). Then, once:

```bash
/mnt/scratch/users/3148123l/venvs/retrievers/bin/hf auth login
```

and accept the licence of [naver/splade-v3](https://huggingface.co/naver/splade-v3) on Hugging Face (it is gated).
Compute nodes reach Hugging Face and ir_datasets through the cluster proxy, so models and corpora download on first use.

## Running

[execute_me.md](execute_me.md) has every command, grouped per retriever. The pattern is:

```bash
sbatch retrievers/slurm/cpu.sbatch --retriever bm25 --dataset beir          # BM25: CPU job, 32 threads
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset beir            # SPLADE/E5/ColBERT: 1 L40S
bash   retrievers/slurm/gpu.sbatch --retriever e5 --dataset beir/scifact    # the same script without SLURM
python retrievers/aggregate.py                                              # summary tables
```

Submit from the repository root (SLURM writes `retrievers/logs/<job>-<id>.out`). The sbatch scripts set
`HF_HOME=/mnt/scratch/users/3148123l/hf_home` and `IR_DATASETS_HOME=/mnt/scratch/users/3148123l/ir_datasets`
(corpora are too large for the 250G home quota); export your own values to change them.

`run.py` options:

| Option | Default | Meaning |
|---|---|---|
| `--retriever` | | `bm25`, `splade`, `e5`, `colbert` |
| `--dataset` | | a name (`beir/scifact`), a group (`beir`, `msmarco`, `bright`, `miracl`, `coir`, `all`), a prefix (`beir/cqadupstack`), or a comma list |
| `--k` | 100 | results per query |
| `--threads` | `$SLURM_CPUS_PER_TASK`, else all cores | PISA indexing/retrieval threads |
| `--model` | per dataset (table above) | another Hugging Face model; results go to `experiments/<retriever>-<model>/` |
| `--batch_size` | 64 SPLADE (32 multilingual), 256 E5, 128 ColBERT | encoder batch size |
| `--plaid_batch` | 50,000 | documents buffered (fp32 token embeddings) per PLAID add |
| `--colbert_query_length`, `--colbert_doc_length` | model defaults (mLateOn docs: 512) | ColBERT truncation |
| `--overwrite_run` | off | recompute runs whose `metrics.json` exists (indexes are always reused) |
| `--keep_going` | off | log a failing dataset and continue with the group |

## Output

```
retrievers/
  index/<retriever>-<model>/<dataset>.{pisa,flex,plaid}   gitignored, reused by later runs
                                                          (the MS MARCO sets share msmarco-passage.<ext>)
  experiments/<retriever>/<dataset>/
      run.trec.gz        top-100 TREC run (rank from 1)
      metrics.json       measures, model, index path, query counts, timings
      perquery.csv       qid, measure, value
  experiments/summary.{md,csv}                            written by aggregate.py
  logs/                                                   gitignored, SLURM output
```

## How it works

For each (retriever, dataset), `run.py`:

1. builds the index if it is not there (`retrievers/index/...`), otherwise loads it;
2. retrieves `k` (+ a small margin, see 3) results for every query;
3. applies the benchmark's filter: BEIR drops a document that is the query itself (ArguAna, Quora); BRIGHT drops
   each query's `excluded_ids`. Then it keeps the top `k` = 100;
4. saves the run and evaluates it with [ir_measures](https://ir-measur.es). Each measure is averaged over every query
   with at least one relevant document; a query with no results counts as 0 (like `trec_eval -c`).

Progress: encoders show one tqdm bar per corpus. PISA never gets a tqdm bar (it breaks PISA's own output); BM25
indexing logs a line every 500k documents and PISA prints its own build steps.

### Retriever details

- **BM25 (PISA).** Repeated query terms count (`query_weighted=True`), as in Lucene; without it ArguAna loses
  0.07 nDCG@10. PISA's tokenizer drops every non-ASCII character, so for the 17 non-English MIRACL languages the
  text is tokenized in Python (NFKC, casefold, Unicode letter/number/mark runs; Chinese, Japanese and Thai runs as
  overlapping character bigrams, like Lucene's CJKAnalyzer) and indexed as term counts with `toks_indexer`. BM25 on
  those counts gives the same scores as a text index (checked on vaswani). There is no stemming for those languages.
  Pyserini's MIRACL baselines use Lucene language analyzers, so expect some per-language differences.
- **SPLADE (PISA).** Documents and queries are encoded on the GPU in fp16, with max-pooled log(1+ReLU) MLM logits
  truncated to 512 tokens. Weights are scaled by 100 and rounded for PISA. The multilingual OpenSearch model encodes
  only documents, with special tokens zeroed and weights under 0.1 × the document maximum pruned, as on its model
  card. Its queries are inference-free: each query token gets its `idf.json` weight. It does not cover Thai.
- **E5 (FlexIndex).** `pyterrier_dr.E5` adds the `query: ` / `passage: ` prefixes and normalises, encoding in fp16.
  Search is exact. It runs on the GPU (fp16) when the index fits in 30 GB (`RB_GPU_INDEX_GB`); otherwise it runs
  numpy brute force on the CPU (MIRACL en, de, fr, es).
- **ColBERT (PLAID).** pyterrier-pylate with fast-plaid at PyLate's defaults (nbits 4). PLAID centroids come from
  the first `--plaid_batch` documents. `colbertv2.0` keeps its 32-token query limit, which truncates long BRIGHT
  queries; pass `--colbert_query_length 128` for a variant.

### Sanity checks (nDCG@10 measured with this code)

| Dataset | Retriever | Here | Reference |
|---|---|---|---|
| BEIR SciFact | BM25 | 0.676 | 0.679 Pyserini BM25-flat |
| BEIR ArguAna | BM25 | 0.419 | 0.397 flat / 0.414 multifield (Pyserini) |
| BRIGHT biology | BM25 | 0.171 | 0.189 (BRIGHT paper, Pyserini) |
| MIRACL sw | BM25 | 0.467 | 0.383 (MIRACL paper; whitespace analyzer) |
| BEIR SciFact | SPLADE++ ED (`--model naver/splade-cocondenser-ensembledistil`) | 0.708 | ≈0.70 (SPLADE++ paper, BEIR table) |
| BEIR SciFact | E5 (e5-base-v2) | 0.719 | ≈0.72 (MTEB SciFact) |
| BEIR SciFact | ColBERT (colbertv2.0) | 0.692 | ≈0.69 (ColBERTv2 paper) |
| MIRACL sw | SPLADE (OpenSearch multilingual) | 0.766 | |
| MIRACL sw | E5 (multilingual-e5-base) | 0.713 | |
| MIRACL sw | ColBERT (mLateOn; sw is not one of its training languages) | 0.574 | |

## Running on a new machine or pod

`gpu.sbatch` prints the GPU name (`nvidia-smi`) and stops at once with `torch sees no GPU` if torch cannot use it,
instead of encoding on the CPU. Known ways a run fails on a fresh pod:

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'ir_measures'` (or any other package) | `setup.sh` failed, leaving an empty venv. Before the split install, one unreachable GitHub requirement (`pyterrier-pylate`) aborted the whole `uv pip install` | re-run `bash retrievers/setup.sh` and check it ends with `venv ready`; do not start jobs if it fails |
| `Failed to connect to github.com port 443` during setup | the pod has no route to GitHub | BM25/SPLADE/E5 still install; for ColBERT open GitHub access, or build a wheel elsewhere (`uv build --wheel git+https://github.com/lightonai/pyterrier-pylate -o wheels/`) and `uv pip install` it |
| `401 ... Cannot access gated repo ... naver/splade-v3` | no Hugging Face login on that machine | `export HF_TOKEN=...` or `$VENV/bin/hf auth login`, with the splade-v3 licence accepted |
| `torch sees no GPU` | the venv's torch is a CUDA 13 build (`+cu130`) and needs a matching driver, or no GPU is attached | `nvidia-smi` on the pod; with an older driver install a torch build for that CUDA version |
| `gpu=` empty in the first log line | no GPU visible to the job (outside SLURM `CUDA_VISIBLE_DEVICES` is usually unset, which is fine) | check `nvidia-smi` |
| permission or "no such directory" errors under `/mnt/scratch/...` | `VENV`, `HF_HOME` and `IR_DATASETS_HOME` default to this cluster's scratch paths | export the three variables to paths that exist and are writable on that machine |

## Scale and caveats

- MIRACL corpora are large (en 32.9M, de 15.9M, fr 14.6M, es 10.4M, ru 9.5M, ja 6.9M, zh 4.9M passages). Give the
  big languages their own jobs (execute_me.md does), and expect E5/ColBERT on MIRACL-en to need most of a day on
  one L40S. ColBERT PLAID indexes for the whole of MIRACL take hundreds of GB.
- MS MARCO dev, DL19 and DL20 query the full 8.8M-passage corpus through one index per retriever
  (`index/<retriever>-<model>/msmarco-passage.<ext>`), built by whichever of the three runs first. Run them in one
  job (`--dataset msmarco`) so two jobs never build it at once. DL qrels are graded 0-3; RR, AP and R count
  grade >= 2 as relevant, the TREC DL convention.
- The English-only models (splade-v3, e5-base-v2, colbertv2.0) are used on CoIR code too, since none of the
  requested models is code-specific.
- OHSUMED is not included. The dataset registry ([rbench/datasets.py](rbench/datasets.py)) takes a new `Bench` in a
  few lines.
