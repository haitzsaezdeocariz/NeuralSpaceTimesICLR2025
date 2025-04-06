#!/bin/bash

for model_type in 'des'
do
    for dataset in 'silico' 'coli' 'cerevisiae'
    do
        for dim in 2 4 10
        do
            echo "Running with configuration: model_type = $model_type, dataset = $dataset, space_dim = $dim, time_dim = 1"
            python train_dream.py --model_type $model_type --device 0 --num_epochs 5000 --display_epoch 5000 --space_dim $dim --time_dim 1 --dataset $dataset
        done
    done
done
