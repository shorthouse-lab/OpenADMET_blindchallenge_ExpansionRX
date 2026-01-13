#!/bin/bash -l
# SGE script: run all OpenADMET methods sequentially on a single node

# Job resources (adjust if needed)
#$ -N openadmet_single
#$ -l h_rt=48:0:0        # walltime reservation
#$ -l mem=4G             # per-core RAM (4G * 4 cores = 16G total)
#$ -pe smp 4             # CPU cores
#$ -l gpu=1              # drop if CPU-only
#$ -cwd
#$ -j y
# Log file; ensure the directory exists before qsub
#$ -o logs/$JOB_ID.log

# Modules for Python 3.9.10 and CUDA 11.8 (adjust module names to your cluster)
module load python/3.9.10
module load cuda/11.8

# Activate local venv
source /home/uccakjo/ACFS/OpenADMET_blindchallenge_ExpansionRX/ChemGridML/.venv/bin/activate

# Optional tuning
export N_JOBS_SKLEARN=4    # sklearn threads; keep <= cores
# export CHEMGRID_FORCE_CPU=1  # uncomment to force CPU

python run_openadmet.py \
  --config openadmet_config.yaml \
  --results-dir openadmet_results_single \
  --force
