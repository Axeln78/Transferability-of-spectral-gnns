#!/bin/bash

FILENAME=out
tmux new -s ogb -d
tmux send-keys "source activate benchmark_gnn" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molhiv' --gnn Cheb_net --filename $FILENAME &
python main_dgl.py --dataset 'ogbg-molpcba' --gnn Cheb_net --filename $FILENAME
wait" C-m
tmux send-keys "tmux kill-session -t benchmark" C-m
