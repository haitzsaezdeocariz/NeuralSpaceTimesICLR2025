#!/bin/bash

for model_type in 'min' 'des'
do
    for dataset in 'Arxiv'
    do
        for dim in 2
        do
            echo "Running with configuration: model_type = $model_type, dataset = $dataset, space_dim = $dim, time_dim = 1"
            python train_real_baseline.py --model_type $model_type --device 0 --num_epochs 40 --display_epoch 1 --space_dim $dim --time_dim 1 --dataset $dataset
        done
    done
done
