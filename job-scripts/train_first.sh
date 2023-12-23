#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0
#SBATCH --mail-user=tianmingwang@uvic.ca
#SBATCH --mail-type=ALL

cd $project/embedding_detection/openai_embedding_content_anomaly_detection
module purge
module load python/3
source tensorflow/bin/activate

python content_classifier.py