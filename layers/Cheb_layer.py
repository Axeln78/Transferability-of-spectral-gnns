import dgl
import dgl.function as fn
import torch
import torch.nn as nn

"""
    Cheb
"""


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, k):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(k * in_feats, out_feats)

    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}


class ChebLayer(nn.Module):
    """
        Param: [in_dim, out_dim, k, activation, dropout, graph_norm, batch_norm, residual connection]
    """

    def __init__(
            self,
            in_dim,
            out_dim,
            k,
            activation,
            dropout,
            graph_norm,
            batch_norm,
            residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self._k = k
        self.linear = nn.Linear(k * in_dim, out_dim, bias=False)

        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.apply_mod = NodeApplyModule(
            in_dim,
            out_dim,
            k=self._k)

    def forward(self, g, feature, lambda_max=None):
        h_in = feature  # to be used for residual connection

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

            # Put the Chebyschev polynomes as featuremaps
            # g.ndata['h'] = Xt
            # g.apply_nodes(func=self.apply_mod)
            # h = g.ndata.pop('h')
            # linear projection
            h = self.linear(Xt)

        # if self.graph_norm:
        #    h = h * snorm_n  # normalize activation w.r.t. graph size

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, k={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self._k)
