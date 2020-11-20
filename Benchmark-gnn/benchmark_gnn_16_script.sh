#!/bin/bash

seed0=41
seed1=95
seed2=12
seed3=35

code=main_molecules_graph_regression.py
dataset=ZINC
tmux new -s benchmark -d
tmux send-keys "conda activate benchmark_gnn" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_Cheb_16_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_Cheb_16_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_Cheb_16_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_Cheb_16_ZINC.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_Cheb_16_ZINC_no_RC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_Cheb_16_ZINC_no_RC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_Cheb_16_ZINC_no_RC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_Cheb_16_ZINC_no_RC.json' &
wait" C-m

code=main_SBMs_node_classification.py
dataset=SBM_PATTERN
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_Cheb_16_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_Cheb_16_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_Cheb_16_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_Cheb_16_PATTERN.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_Cheb_16_PATTERN_no_RC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_Cheb_16_PATTERN_no_RC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_Cheb_16_PATTERN_no_RC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_Cheb_16_PATTERN_no_RC.json' &
wait" C-m

code=main_SBMs_node_classification.py
dataset=SBM_CLUSTER
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_Cheb_16_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_Cheb_16_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_Cheb_16_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_Cheb_16_CLUSTER.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_Cheb_16_CLUSTER_no_RC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_Cheb_16_CLUSTER_no_RC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_Cheb_16_CLUSTER_no_RC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_Cheb_16_CLUSTER_no_RC.json' &
wait" C-m


tmux send-keys "tmux kill-session -t benchmark" C-m
