#!/bin/bash

for i in {1..5}
do
  for num_nodes in 50
  do
    for dim in 2 4 10
    do
      echo "Running with configuration: distance_type=type$i, num_nodes=$num_nodes, space_dim=$dim, time_dim=$dim"
      python train_dag.py --device 0 --distance_type "type$i" --num_epochs 5000 --num_nodes $num_nodes --seed 123 --space_dim $dim --time_dim $dim
    done
  done
done
