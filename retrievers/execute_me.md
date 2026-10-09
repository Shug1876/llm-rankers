# execute_me

Every command to produce the full benchmark, grouped per retriever. Run them **from the repository root**.
Each line is one SLURM job; jobs are independent and can be queued together. A dataset whose
`experiments/<retriever>/<dataset>/metrics.json` already exists is skipped, so re-submitting a group after a crash
or a timeout only redoes what is missing (indexes are reused too).

To run any line without SLURM (e.g. on a node you already hold with `srun --pty`), replace `sbatch` with `bash`.

## 0. Once

```bash
bash retrievers/setup.sh
/mnt/scratch/users/3148123l/venvs/retrievers/bin/hf auth login     # then accept https://huggingface.co/naver/splade-v3
```

Smoke test (a few minutes; checks each backend end to end):

```bash
bash retrievers/slurm/cpu.sbatch --retriever bm25 --dataset beir/scifact --threads 8
sbatch retrievers/slurm/gpu.sbatch --retriever splade  --dataset beir/scifact,miracl/sw
sbatch retrievers/slurm/gpu.sbatch --retriever e5      --dataset beir/scifact,miracl/sw
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset beir/scifact,miracl/sw
```

Expected nDCG@10 (measured with this code): SciFact BM25 0.676, E5 0.719, ColBERT 0.692; MIRACL-sw BM25 0.467,
SPLADE (OpenSearch multilingual) 0.766, mE5 0.713, mLateOn 0.574. SPLADE-v3 on SciFact was not run here (gated model; ≈0.70 published).

Run BM25 first: it downloads every corpus to `IR_DATASETS_HOME`, so the GPU jobs afterwards do not race each
other on the same download.

## 1. BM25 (PISA, CPU, 32 threads)

```bash
sbatch retrievers/slurm/cpu.sbatch --retriever bm25 --dataset beir --keep_going
sbatch retrievers/slurm/cpu.sbatch --retriever bm25 --dataset msmarco
sbatch retrievers/slurm/cpu.sbatch --retriever bm25 --dataset bright --keep_going
sbatch retrievers/slurm/cpu.sbatch --retriever bm25 --dataset coir --keep_going
sbatch retrievers/slurm/cpu.sbatch --retriever bm25 --dataset miracl --keep_going
```

## 2. SPLADE (splade-v3 / OpenSearch multilingual; GPU encoding, PISA)

```bash
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset beir --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset msmarco
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset bright --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset coir --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset miracl/en
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset miracl/de,miracl/fr
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset miracl/es,miracl/ru
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset miracl/ja,miracl/zh,miracl/ar,miracl/fa,miracl/fi --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever splade --dataset miracl/bn,miracl/hi,miracl/id,miracl/ko,miracl/sw,miracl/te,miracl/th,miracl/yo --keep_going
```

## 3. E5 (e5-base-v2 / multilingual-e5-base; GPU encoding, exact search)

```bash
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset beir --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset msmarco
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset bright --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset coir --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset miracl/en
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset miracl/de,miracl/fr
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset miracl/es,miracl/ru
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset miracl/ja,miracl/zh,miracl/ar,miracl/fa,miracl/fi --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset miracl/bn,miracl/hi,miracl/id,miracl/ko,miracl/sw,miracl/te,miracl/th,miracl/yo --keep_going
```

## 4. ColBERT (colbertv2.0 / mLateOn; GPU encoding, PLAID)

```bash
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset beir --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset msmarco
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset bright --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset coir --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset miracl/en
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset miracl/de
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset miracl/fr
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset miracl/es,miracl/ru
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset miracl/ja,miracl/zh,miracl/ar,miracl/fa,miracl/fi --keep_going
sbatch retrievers/slurm/gpu.sbatch --retriever colbert --dataset miracl/bn,miracl/hi,miracl/id,miracl/ko,miracl/sw,miracl/te,miracl/th,miracl/yo --keep_going
```

mLateOn on the large MIRACL languages is the heaviest job in this file (512-token documents). If a job hits the
3-day limit, re-submit the same line; finished datasets are skipped, but an index that was half built is rebuilt.
`--partition=gpu-h100 --gres=gpu:h100:1` before the script name moves a job to an H100.

## 5. Summary

```bash
/mnt/scratch/users/3148123l/venvs/retrievers/bin/python retrievers/aggregate.py
```

Writes `retrievers/experiments/summary.md` (one table per benchmark and measure; BEIR with CQADupStack averaged
into one row, CoIR with CodeSearchNet and CodeSearchNet-CCR each averaged into one task) and `summary.csv`.

## Monitoring

```bash
squeue -u $USER
tail -f retrievers/logs/rb-gpu-<jobid>.out
grep -h "\[done\]" retrievers/logs/*.out
```
