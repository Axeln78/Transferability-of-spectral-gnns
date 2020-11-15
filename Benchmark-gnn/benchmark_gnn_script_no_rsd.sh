#!/bin/bash

seed0=41
seed1=95
seed2=12
seed3=35

tmux new -s benchmark -d
tmux send-keys "conda activate benchmark_gnn" C-m

code=main_SBMs_node_classification.py
dataset=SBM_PATTERN
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_Cheb_PATTERN_no_rsd.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_Cheb_PATTERN_no_rsd.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_Cheb_PATTERN_no_rsd.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_Cheb_PATTERN_no_rsd.json' &
wait" C-m

code=main_SBMs_node_classification.py
dataset=SBM_CLUSTER
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_Cheb_CLUSTER_no_rsd.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_Cheb_CLUSTER_no_rsd.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_Cheb_CLUSTER_no_rsd.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_Cheb_CLUSTER_no_rsd.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark" C-m
