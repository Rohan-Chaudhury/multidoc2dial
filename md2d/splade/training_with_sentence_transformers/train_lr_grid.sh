#!/usr/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --mail-user=adityasv@andrew.cmu.edu
#SBATCH --mail-type=ALL


export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 

for i in {1..10}; do
    echo "LR: " ${i}e-6
    python splade_md2d.py \
        --model_name /home/adityasv/multidoc2dial/splade/weights/splade_max \
        --use_all_queries \
        --data_path /home/adityasv/multidoc2dial/data/mdd_dpr/beir_format \
        --lr ${i}e-6 \
        --epochs 10 \
        --use_all_queries
done