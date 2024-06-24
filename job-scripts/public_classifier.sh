#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:0:0
#SBATCH --mail-user=wangmaine@gmail.com
#SBATCH --mail-type=ALL

cd $project/embedding_detection/openai_embedding_content_anomaly_detection
module purge
module load python/3.11 scipy-stack
# source ./pyscipy/bin/activate
source tensorflow/bin/activate

python content_classifier_iot23.py