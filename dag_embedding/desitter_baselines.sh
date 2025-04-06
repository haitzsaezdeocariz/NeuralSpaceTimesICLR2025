#!/bin/bash

for i in {1..5}
do
  for num_nodes in 50
  do
    for dim in 2 4 10
    do
      echo "Running with configuration: model_type='des', distance_type=type$i, num_nodes=$num_nodes, space_dim=$dim, time_dim=1"
      python train_dag.py --model_type 'des' --device 0 --distance_type "type$i" --num_epochs 5000 --display_epoch 5000 --num_nodes $num_nodes --seed 123 --space_dim $dim --time_dim 1
    done
  done
done