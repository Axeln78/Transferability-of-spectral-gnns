"""
    File to load dataset based on user control from main file
"""
# from data.COLLAB import COLLABDataset

from data.SBMs import SBMsDataset
from data.molecules import MoleculeDataset
from data.ogb_mol import MolDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC' or DATASET_NAME == 'ZINC-full':
        return MoleculeDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS:
        return SBMsDataset(DATASET_NAME)

    # handling for COLLAB dataset
    #    if DATASET_NAME == 'OGBL-COLLAB':
    #        return COLLABDataset(DATASET_NAME)
    # OGB
    if DATASET_NAME == 'ogbg-molhiv' or 'ogbg-molpcba':
        return MolDataset(DATASET_NAME)
