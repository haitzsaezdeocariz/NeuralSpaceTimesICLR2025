#!/bin/bash

for model_type in 'min' 'des'
do
    for dataset in 'Cornell' 'Texas' 'Wisconsin'
    do
        for dim in 2 4 10
        do
            echo "Running with configuration: model_type = $model_type, dataset = $dataset, space_dim = $dim, time_dim = 1"
            python train_real_baseline.py --model_type $model_type --device 0 --num_epochs 5000 --display_epoch 1000 --space_dim $dim --time_dim 1 --dataset $dataset
        done
    done
done
