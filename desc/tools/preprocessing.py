import numpy as np
import pandas as pd
from scipy.sparse import issparse
from anndata import AnnData


def _log1p(x):
    if issparse(x):
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(x, out=x)
    return x


def log1p(data, copy=False):
    if copy:
        data = data.copy()
    if isinstance(data, AnnData):
        _log1p(data.X)
    else:
        _log1p(data)
    return data if copy else None

def get_mean_var(X):
# - using sklearn.StandardScaler throws an error related to
#   int to long trafo for very large matrices
# - using X.multiply is slower
    if True:
        mean = X.mean(axis=0)
        if issparse(X):
            mean_sq = X.multiply(X).mean(axis=0)
            mean = mean.A1
            mean_sq = mean_sq.A1
        else:
            mean_sq = np.multiply(X, X).mean(axis=0)
        # enforece R convention (unbiased estimator) for variance
        var = (mean_sq - mean**2) * (X.shape[0]/(X.shape[0]-1))
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False).partial_fit(X)
        mean = scaler.mean_
        # enforce R convention (unbiased estimator)
        var = scaler.var_ * (X.shape[0]/(X.shape[0]-1))
    return mean, var


def scale_bygroup(adata,groupby,max_value=6):
    res=None
    assert isinstance(adata,AnnData),'adata must be AnnData class'
    adata.X=adata.X.toarray() if issparse(adata.X) else adata.X
    if groupby in adata.obs.keys():
        df=pd.Series(adata.obs[groupby],dtype="category")
        for category in df.cat.categories:
            tmp=adata[df==category]
            tmp=tmp.X
            tmp=np.asarray(tmp)
            mean0,var0=get_mean_var(tmp)
            sd0=np.sqrt(var0)
            sd0[sd0<=1e-5]=1e-5
            tmp-=mean0
            tmp/=sd0
            if max_value is not None:
                tmp[tmp>max_value]=max_value
            adata.X[df==category]=tmp.copy()
    else:
        print("Warning: The groupby:"+str(groupby)+ "you provided is not exists, we scale across all cells")
        res=adata.X
        #avoid all 0 columns
        mean0,var0=get_mean_var(res)
        sd0=np.sqrt(var0)
        sd0[sd0<=1e-5]=1e-5
        if issparse(res):
            res=res.toarray()
        res-=mean0
        res/=sd0
        res[res>max_value]=max_value
        adata.X=res
    return adata

