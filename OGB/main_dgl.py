import argparse

import dgl
import numpy as np
import torch
import torch.optim as optim
from gnn_dgl import GNN
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, (batch_graphs, batch_labels) in enumerate(tqdm(loader, desc="Iteration")):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_h = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)

        pred = model(batch_graphs, batch_h, batch_e)

        optimizer.zero_grad()

        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch_labels == batch_labels
        if "classification" in task_type:
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
        else:
            loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])

        loss.backward()
        optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (batch_graphs, batch_labels) in enumerate(tqdm(loader, desc="Iteration")):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_h = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)

        with torch.no_grad():
            pred = model(batch_graphs, batch_h, batch_e)

        y_true.append(batch_labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)
    return batched_graph, labels


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with DGL')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gated-gcn',
                        help='GNN (default: gated-gcn)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = DglGraphPropPredDataset(name=args.dataset)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_dgl)

    if args.gnn in ['gated-gcn', 'mlp']:
        model = GNN(gnn_type=args.gnn, num_tasks=dataset.num_tasks, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, dropout=args.dropout, batch_norm=True,
                    residual=True, graph_pooling="mean")
        model.to(device)
    elif args.gnn == 'Cheb_net':
        model = GNN(gnn_type='Cheb_net', num_tasks=dataset.num_tasks, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, dropout=args.dropout, batch_norm=True,
                    residual=True, graph_pooling="mean")
        model.to(device)
    else:
        raise ValueError('Invalid GNN type')

    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({
            'Val': valid_curve[best_val_epoch],
            'Test': test_curve[best_val_epoch],
            'Train': train_curve[best_val_epoch],
            'BestTrain': best_train
        }, args.filename)


if __name__ == "__main__":
    main()
