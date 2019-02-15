
from ..datasets import pbmc_processed
from ..models.desc import train


def run_desc_test():
    print('Start to run a package test!')
    adata = pbmc_processed()
    adata = train(adata, dims=[100, 64, 16], louvain_resolution=0.1)
    return None
