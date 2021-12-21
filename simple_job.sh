#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100l:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=30:00:00
#SBATCH --output=/home/iliaskar/projects/def-mseltzer/iliaskar/model-based-atari/slurm_logs/log_%j_%a.txt
#SBATCH --mail-user=iliaskarimalis@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mseltzer


cd /home/iliaskar/projects/def-mseltzer/iliaskar/model-based-atari/SimPLe/

module load python/3.8.10 scipy-stack

pip3 install wandb

python3 -m simple --world-model-steps 10000 --agent-steps 800 --policy-optimizer reinforce --trust-region-beta 0.0001 --rollout-length 100 --agents 8 --use-wandb

cd ..
