#!/bin/bash

for dataset in 'Arxiv'
do
    for dim in 2
    do
        echo "Running with configuration: dataset = $dataset, space_dim=$dim, time_dim=$dim"
        python train_real.py --batch_size 10000 --device 0 --num_epochs 40 --display_epoch 1 --space_dim $dim --time_dim $dim --dataset $dataset
    done
done
