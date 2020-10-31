"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.molecules_graph_regression.gcn_net import GCNNet
from nets.molecules_graph_regression.ChebNet import ChebNet

def GCN(net_params):
    return GCNNet(net_params)


def Cheb(net_params):
    return ChebNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'ChebNet': Cheb
    }

    return models[MODEL_NAME](net_params)
