#!/bin/bash

tmux new -s ogb -d
tmux send-keys "conda activate benchmark_gnn" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_1 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_2 --num_workers 1 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_3 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_4 --num_workers 1 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_5 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_6 --num_workers 1 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_7 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_8 --num_workers 1 &
wait" C-m
tmux send-keys "
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_9 --num_workers 1 &
python main_dgl.py --dataset 'ogbg-molhiv' --batch_size 128 --gnn Cheb_net --filename out__hiv_0 --num_workers 1 &
wait" C-m

tmux send-keys "tmux kill-session -t ogb" C-m
