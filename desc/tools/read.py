from anndata import read_mtx
from anndata.utils import make_index_unique
import pandas as pd


def read_10X(data_path, var_names='gene_symbols'):

    adata = read_mtx(data_path + '/matrix.mtx').T

    genes = pd.read_csv(data_path + '/genes.tsv', header=None, sep='\t')
    adata.var['gene_ids'] = genes[0].values
    adata.var['gene_symbols'] = genes[1].values

    assert var_names == 'gene_symbols' or var_names == 'gene_ids', \
        'var_names must be "gene_symbols" or "gene_ids"'

    if var_names == 'gene_symbols':
        var_names = genes[1]
    else:
        var_names = genes[0]

    if not var_names.is_unique:
        var_names = make_index_unique(pd.Index(var_names))
        print('var_names are not unique, "make_index_unique" has applied')

    adata.var_names = var_names

    cells = pd.read_csv(data_path + '/barcodes.tsv', header=None, sep='\t')
    adata.obs['barcode'] = cells[0].values
    adata.obs_names = cells[0]
    return adata
