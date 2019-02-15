"""
from MulticoreTSNE import MulticoreTSNE as mTSNE
from anndata import AnnData
import matplotlib.pyplot as plt

def run_tsne(
        data,
        return_tsne=True,
        tsne_plot=True,
        n_jobs=1,
        n_components=2,
        perplexity=40,
        learning_rate=200
        ):
    assert isinstance(data, AnnData), 'Required input should be an AnnData object from desc'
    assert hasattr(data.obsm, 'embedded'), 'The embedded matrix not found, run desc first'
    tsne = mTSNE(n_jobs=n_jobs, n_components=n_components, perplexity=perplexity, learning_rate=learning_rate). \
        fit_transform(data.obsm['embedded'])
    data.obsm['tsne'] = tsne

    if tsne_plot:
        plt.scatter(tsne[:, 0], tsne[:, 1], s=0.1, c=data.obs['ident'])
        plt.title("tSNE plot")
        plt.xlabel("tSNE_1")
        plt.ylabel("tSNE_2")

    return tsne if return_tsne else None
"""
