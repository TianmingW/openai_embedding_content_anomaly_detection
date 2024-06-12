#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:0:0
#SBATCH --mail-user=wangmaine@gmail.com
#SBATCH --mail-type=ALL

cd $project/embedding_detection/openai_embedding_content_anomaly_detection
# module purge
module load python/3.11 scipy-stack
source ./py311/bin/activate
# source ./pyscipy/bin/activate

python data_preprocess_iot23.py