#!/bin/bash

for i in 1
do
  for num_nodes in 50
  do
    for dim in 2 4 10
    do
      for p in 50 10
      do
        echo "Running with configuration: distance_type=type$i, num_nodes=$num_nodes, space_dim=$dim, time_dim=$dim, p=$p"
        python train_dag.py --display_epoch 5000 --device 0 --distance_type "type$i" --num_epochs 5000 --num_nodes $num_nodes --seed 123 --space_dim $dim --time_dim $dim --p $p
      done
    done
  done
done


for i in 1
do
  for num_nodes in 50
  do
    for dim in 2 4 10
    do
      for p in 50 10
      do
        echo "Running with configuration: model_type='min', distance_type=type$i, num_nodes=$num_nodes, space_dim=$dim, time_dim=1, p=$p"
        python train_dag.py --display_epoch 5000 --model_type 'min' --device 0 --distance_type "type$i" --num_epochs 5000 --num_nodes $num_nodes --seed 123 --space_dim $dim --time_dim 1 --p $p
      done
    done
  done
done


for i in 1
do
  for num_nodes in 50
  do
    for dim in 2 4 10
    do
      for p in 50 10
      do
        echo "Running with configuration: model_type='min', distance_type=type$i, num_nodes=$num_nodes, space_dim=$dim, time_dim=1, p=$p"
        python train_dag.py --display_epoch 5000 --model_type 'des' --device 0 --distance_type "type$i" --num_epochs 5000 --num_nodes $num_nodes --seed 123 --space_dim $dim --time_dim 1 --p $p
      done
    done
  done
done
