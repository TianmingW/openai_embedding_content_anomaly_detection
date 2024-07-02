#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:0:0
#SBATCH --mail-user=wangmaine@gmail.com
#SBATCH --mail-type=ALL

cd $project/embedding_detection/openai_embedding_content_anomaly_detection
module purge
module load python/3.11 scipy-stack
# source ./pyscipy/bin/activate
source tensorflow/bin/activate

# Number of runs you want to execute
N=5

for i in $(seq 1 $N)
do
    echo "Running iteration $i"
    python content_classifier_iot23.py $i
done