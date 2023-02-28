#!/usr/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --mail-user=adityasv@andrew.cmu.edu
#SBATCH --mail-type=ALL


export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 

python splade_md2d.py \
    --model_name /home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/splade/distilsplade_max \
    --use_all_queries \
    --data_path /home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/beir_format \
    --lr 1e-6 \
    --epochs 20 \
    --use_all_queries \
    --pair_lambda 0.1


python splade_md2d.py \
    --model_name /home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/splade/splade_distil_CoCodenser_large \
    --use_all_queries \
    --data_path /home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/beir_format \
    --lr 1e-6 \
    --epochs 20 \
    --use_all_queries \
    --pair_lambda 0.1



python splade_md2d.py \
    --model_name /home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/splade/splade_max_CoCodenser \
    --use_all_queries \
    --data_path /home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/beir_format \
    --lr 1e-6 \
    --epochs 30 \
    --use_all_queries \
    --pair_lambda 0.1