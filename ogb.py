import time
import dgl
import torch
from torch.utils.data import Dataset
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator


class MOLHIVataset(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglGraphPropPredDataset(name="ogbg-molhiv")
        self.split_idx = self.dataset.get_idx_split()

        self.evaluator = Evaluator(name='ogb-molhiv')

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

class MOLPCBAataset(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglGraphPropPredDataset(name='ogb-molpcba')

        self.split_idx = self.dataset.get_idx_split()

        self.evaluator = Evaluator(name='ogb-molpcba')

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))