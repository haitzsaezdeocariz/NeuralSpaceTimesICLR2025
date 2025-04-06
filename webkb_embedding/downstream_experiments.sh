#!/bin/bash

# Arguments to iterate over
models=('MLP' 'GCN' 'GAT' 'CPGNNMLP' 'CPGNNCheby')
hidden_dims=(10 20 30 64)
datasets=("Cornell" "Texas" "Wisconsin")
preprocess_values=(true false)  # Handle the flag logic explicitly
 
# Loop over all combinations
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for hidden_dim in "${hidden_dims[@]}"; do
            for preprocess in "${preprocess_values[@]}"; do
                echo "Running with dataset=$dataset, model_type=$model, hidden_dim=$hidden_dim, preprocess=$preprocess"
                
                if [ "$preprocess" == "true" ]; then
                    python node_classification.py --preprocess --model_type "$model" --hidden_dim "$hidden_dim" --datasetname "$dataset"
                else
                    python node_classification.py --model_type "$model" --hidden_dim "$hidden_dim" --datasetname "$dataset"
                fi
            done
        done
    done
done
