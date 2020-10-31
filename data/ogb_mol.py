import time

from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from torch.utils.data import Dataset


class MolDataset(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglGraphPropPredDataset(name=name)
        self.split_idx = self.dataset.get_idx_split()

        self.evaluator = Evaluator(name=name)

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

