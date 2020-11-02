import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)

        self.bn_node_h = nn.BatchNorm1d(output_dim)

        if dropout != 0.0:
            self.drop_h = nn.Dropout(dropout)

    def forward(self, g, h, e):

        h_in = h  # for residual connection

        h = self.A(h)

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization

        h = F.relu(h)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection

        if self.dropout != 0:
            h = self.drop_h(h)  # dropout

        return h, e


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GNN(nn.Module):

    def __init__(self, gnn_type, num_tasks, num_layer=4, emb_dim=256,
                 dropout=0.0, batch_norm=True,
                 residual=True, graph_pooling="mean"):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.graph_pooling = graph_pooling

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        gnn_layer = {
            'Cheb_net': ChebLayer,
            'mlp': MLPLayer,
        }.get(gnn_type, GatedGCNLayer)

        self.layers = nn.ModuleList([
            gnn_layer(emb_dim, emb_dim, dropout=dropout, batch_norm=batch_norm, residual=residual)
            for _ in range(num_layer)
        ])

        self.pooler = {
            "mean": dgl.mean_nodes,
            "sum": dgl.sum_nodes,
            "max": dgl.max_nodes,
        }.get(graph_pooling, dgl.mean_nodes)

        self.graph_pred_linear = MLPReadout(emb_dim, num_tasks)

    def forward(self, g, h, e):
        h = self.atom_encoder(h)
        e = self.bond_encoder(e)

        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        hg = self.pooler(g, 'h')

        return self.graph_pred_linear(hg)


class ChebLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, input_dim, output_dim, dropout=0.0, graph_norm=True, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        # self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self._k = 3

        if self.in_channels != self.out_channels:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(self.out_channels)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.apply_mod = NodeApplyModule(
            self.in_channels,
            self.out_channels,
            k=self._k)
        self.linear = nn.Linear(self._k * self.in_channels, self.out_channels, bias=False)

    # def forward(self, g, feature, snorm_n, lambda_max=None):
    def forward(self, g, feature, e):
        h_in = feature  # to be used for residual connection
        lambda_max = [2] * g.batch_size

        def unnLaplacian(feature, D_sqrt, graph):
            """ Operation D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feature * D_sqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return graph.ndata.pop('h') * D_sqrt

        with g.local_scope():
            D_sqrt = torch.pow(g.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feature.device)

            if lambda_max is None:
                try:
                    lambda_max = dgl.laplacian_lambda_max(g)
                except BaseException:
                    # if the largest eigonvalue is not found
                    lambda_max = [2]

            if isinstance(lambda_max, list):
                lambda_max = torch.Tensor(lambda_max).to(feature.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            lambda_max = dgl.broadcast_nodes(g, lambda_max)

            # X_0(f)
            Xt = X_0 = feature

            # X_1(f)
            if self._k > 1:
                re_norm = (2. / lambda_max).to(feature.device)
                h = unnLaplacian(X_0, D_sqrt, g)
                # print('h',h,'norm',re_norm,'X0',X_0)
                X_1 = - re_norm * h + X_0 * (re_norm - 1)

                Xt = torch.cat((Xt, X_1), 1)

            # Xi(x), i = 2...k
            for _ in range(2, self._k):
                h = unnLaplacian(X_1, D_sqrt, g)
                X_i = - 2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0

                Xt = torch.cat((Xt, X_i), 1)
                X_1, X_0 = X_i, X_1

            h = self.linear(Xt)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h, e
