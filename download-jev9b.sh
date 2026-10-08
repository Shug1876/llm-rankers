#!/usr/bin/env bash

hf download autotrust/JEV-9B --include "vl/*" --local-dir JEV-9B
bash JEV-9B/vl/serve.sh     # downloads Qwen/Qwen3.5-9B (with its vision encoder) and serves both systems on :8000
