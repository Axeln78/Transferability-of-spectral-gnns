#!/bin/bash

FILENAME=out
tmux new -s ogb -d
tmux send-keys "conda activate benchmark_gnn" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_1 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_2 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_3 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_4 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_5 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_6 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_7 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_8 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 128 --gnn Cheb_net --filename out_9 &
wait" C-m
tmux send-keys "tmux kill-session -t ogb" C-m
