#!/bin/bash


tmux new -s ogb -d
tmux send-keys "conda activate benchmark_gnn" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_1 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_2 --num_workers 1 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_3 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_4 --num_workers 1 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_5 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_6 --num_workers 1 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_7 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_8 --num_workers 1 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_9 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molpcba' --batch_size 256 --gnn Cheb_net --filename out_10 --num_workers 1 &
wait" C-m

tmux send-keys "tmux kill-session -t ogb" C-m
