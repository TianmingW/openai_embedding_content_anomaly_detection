#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:0:0
#SBATCH --mail-user=tianmingwang@uvic.ca
#SBATCH --mail-type=ALL

cd~/$project/embedding_detection/openai_embedding_content_anomaly_detection
module purge
module load python/3.11 scipy-stack
source ./py310/bin/activate

python helloworld.py