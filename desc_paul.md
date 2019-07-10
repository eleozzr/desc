
# The Tutorial of Paul et al (2015) for [desc](https://eleozzr.github.io/desc)
- The reproducible of DESC for Paul's data


```python
import os              
os.environ['PYTHONHASHSEED'] = '0'
import desc          
import pandas as pd                                                    
import numpy as np                                                     
import scanpy.api assc                                                                                 
from time import time                                                       
import sys
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline 
sc.settings.set_figure_params(dpi=300)
```
Using TensorFlow backend.

```python
print(sys.version)
```
 3.5.3 |Continuum Analytics, Inc.| (default, Mar  6 2017, 11:58:13) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]

```python
sc.logging.print_versions()
```
scanpy==1.3.6 anndata==0.6.18 numpy==1.14.6 scipy==1.1.0 pandas==0.23.0 scikit-learn==0.19.2 statsmodels==0.9.0 python-igraph==0.7.1 louvain==0.6.1 



```python
desc.__version__
```
'1.0.0.2'
```python
#we have downloaded this data, if not, the following will download data automatically
adata=sc.datasets.paul15()
adata.obs['celltype']=adata.obs['paul15_clusters'].str.split("[0-9]{1,2}", n = 1, expand = True).values[:,1]
adata.obs['celltype2']=adata.obs['paul15_clusters']
```

WARNING: In Scanpy 0.*, this returned logarithmized data. Now it returns non-logarithmized data.
 ... storing 'paul15_clusters' as categorical

```python
import desc
```


```python
sc.pp.log1p(adata)
sc.pp.filter_genes_dispersion(adata,n_top_genes=1000)
sc.pp.scale(adata,max_value=6)
save_dir="paul_result"
adata=desc.train(adata,
        dims=[adata.shape[1],64,32],
        tol=0.005,
        n_neighbors=10,
        batch_size=256,
        louvain_resolution=[0.8,1.0],# not necessarily a list, you can only set one value, like, louvain_resolution=1.0
        save_dir=str(save_dir),
        do_tsne=True,
        learning_rate=200, # the parameter of tsne
        use_GPU=False,
        num_Cores=1, #for reproducible, only use 1 cpu
        num_Cores_tsne=4,
        save_encoder_weights=False,
        save_encoder_step=3,# save_encoder_weights is False, this parameter is not used
        use_ae_weights=False,
        do_umap=False) #if do_uamp is False, it will don't compute umap coordiate
```

```python
#After training from desc, the results have been saved in `save_dir`
#adata=sc.read("paul_result/adata_desc.h5ad")
adata
```
AnnData object with n_obs × n_vars = 2730 × 1000 
   obs: 'paul15_clusters', 'celltype', 'celltype2', 'desc_0.8', 'desc_1.0'
   var: 'means', 'dispersions', 'dispersions_norm'
   uns: 'prob_matrix1.0', 'prob_matrix0.8', 'iroot'
   obsm: 'X_Embeded_z0.8', 'X_tsne', 'X_tsne0.8', 'X_Embeded_z1.0', 'X_tsne1.0'

1. The meta.data of each cell has been saved in `adata.obs`, 
2. the representation from `desc` of each cell have been saved in `adata.obsm`('X_Embeded_z1.0')
3. The dimension reduction from `desc` of each cell have beed saved in `adata.obsm`('X_tsne1.0')

# Computing maxmum probability
```python
adata.obs['max.prob']=adata.uns["prob_matrix1.0"].max(1)
```
# t-SNE plots
```python
sc.pl.scatter(adata,basis="tsne1.0",color=['desc_1.0','max.prob'])
```
![png](output_13_1.png)

```python
sc.pl.scatter(adata,basis="tsne1.0",color=['celltype','celltype2'])
```
![png](output_14_0.png)


# reference

1. Paul F, Arkin Y, Giladi A, Jaitin DA et al. Transcriptional heterogeneity and lineage commitment in myeloid progenitors. Cell, 163:1663-167 (2015)
