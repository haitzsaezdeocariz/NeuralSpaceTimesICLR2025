#!/bin/bash

for dataset in 'silico' 'coli' 'cerevisiae'
do
    for dim in 2 4 10
    do
        echo "Running with configuration: dataset = $dataset, space_dim=$dim, time_dim=$dim"
        python train_dream.py --device 0 --num_epochs 5000 --display_epoch 1000 --space_dim $dim --time_dim $dim --dataset $dataset
    done
done

