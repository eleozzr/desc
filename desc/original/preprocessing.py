import scanpy.api as sc
import pandas as pd
import numpy as np
import scipy
import os
from scipy.sparse import issparse
import datetime 

def read_dataset(input_file,transpose=False):
    """
    Construct a anndata object
       
    """
    if os.path.isfile(input_file):
        print("The value os", os.path.isfile(input_file))
        if str(input_file).endswith('h5ad'):
            adata=sc.read(input_file)
        elif sum([str(input_file).endswith(str(i)) for i in ["tsv",'TSV','tab','data']]):
            adata=sc.read_text(input_file,sep="\t",first_column_names=True)
            if transpose:
                adata=adata.T
        elif sum([str(input_file).endswith(str(i)) for i in ['csv',"CSV"]]):
            adata=sc.read_text(input_file,sep=",",first_column_names=True) 
            if transpose:
                adata=adata.T
        else:
            #ValueError 'The file must be one of *.h5ad, *.tsv,*TSV,*.tab,*data, *csv,*CSV' 
            print("The file must be one of *.h5ad, *.tsv,*TSV,*.tab,*data, *csv,*CSV")
    else:
        #read folder
        mtx=sc.read_mtx(os.path.join(input_file,"matrix.mtx")) 
        num_lines = sum(1 for line in open(os.path.join(input_file,'barcodes.tsv')))
        cellinfo=pd.read_csv(os.path.join(input_file,"barcodes.tsv"),sep="\t",header=None if num_lines==mtx.shape[1] else 0)
        if not 'cellname' in cellinfo.columns:
            cellinfo['cellname']=cellinfo.iloc[:,0]
        num_lines = sum(1 for line in open(os.path.join(input_file,'genes.tsv')))
        geneinfo=pd.read_csv(os.path.join(input_file,"genes.tsv"),sep="\t",header=None if num_lines==mtx.shape[0] else 0)
        if not 'genename' in geneinfo.columns:
            geneinfo['genename']=geneinfo.iloc[:,1]# for 10x,the second columns is the genename, and the first column is gene_id
        #create anndata
        adata=sc.AnnData(mtx.X.T,obs=cellinfo,var=geneinfo)
        adata.obs_names=adata.obs["cellname"]
        adata.var_names=adata.var["genename"]
        adata.obs_names_make_unique(join="-")
        adata.var_names_make_unique(join="-")
    #create time
    now = datetime.datetime.now()
    adata.uns["ProjectName"]="DESC created in"+str(now.strftime("%Y-%m-%d %H:%M")) 
    print("Creat adata successfully! The adata infofation is", adata)
    return adata

        
def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw=sc.pp.log1p(adata,copy=True) #check the rowname 
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)
   

    
        
def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)
    sc.pp.filter_cells(adata,min_genes=1)#avoiding some cell all 0




def filter_cells_mito(adata, mito_percent_cutoff=0.10,nUMIbefore_filter_MT_gene=True):
    mito_genes=[name.startswith("MT-") for name in adata.var_names]
    if sum(mito_genes)>0:
        if issparse(adata.X):
            adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
        else:
            adata.obs['percent_mito']=np.sum(adata[:,mito_genes].X,axis=1)/np.sum(adata.X,axis=1)
    if 'percent_mito' in adata.obs.columns:
        if nUMIbefore_filter_MT_gene:
            if issparse(adata.X): 
                adata.obs['n_counts_before']=adata.X.sum(axis=1).A1
                adata.obs['n_genes_before']=(adata.X!=0).sum(axis=1).A1
            else:
                adata.obs['n_counts_before']=adata.X.sum(axis=1)
                adata.obs['n_genes_before']=(adata.X!=0).sum(axis=1)
            adata._inplace_subset_obs(adata.obs['percent_mito'] < mito_percent_cutoff)
            adata.inplace_subset_var(~np.asarray(mito_genes))
        else:
            adata._inplace_subset_var(~np.asarray(mito_genes))
            adata._inplace_subset_obs(adata.obs['percent_mito'] < mito_percent_cutoff)
            if issparse(adata.X): 
                adata.obs['n_counts_after']=adata.X.sum(axis=1).A1
                adata.obs['n_genes_after']=(adata.X!=0).sum(axis=1).A1
            else:
                adata.obs['n_counts_after']=adata.X.sum(axis=1)
                adata.obs['n_genes_after']=(adata.X!=0).sum(axis=1)
    else:
        if issparse(adata.X): 
            adata.obs['n_counts']=adata.X.sum(axis=1).A1
            adata.obs['n_genes']=(adata.X!=0).sum(axis=1).A1
        else:
            adata.obs['n_counts']=adata.X.sum(axis=1)
            adata.obs['n_genes']=(adata.X!=0).sum(axis=1)
        
def scale_by_group(adata,groupby,scale_data=0,max_value=6):
    if scale_data==0: return adata
    if scale_data==1: return sc.pp.scale(adata,zero_center=True, max_value=max_value,copy=True)
    if groupby in adata.obs.keys():
        df=pd.Series(adata.obs[groupby],dtype="category")
        for category in df.cat.categories:
            tmp=adata[df==category].X
            mean0,var0=get_mean_var(tmp)
            sd0=np.sqrt(var0)
            sd0[sd0<=1e-5]=1e-5
            if issparse(tmp):
                tmp=tmp.toarray()
            tmp-=mean0
            tmp/=sd0
            if max_value is not None:
                tmp[tmp>max_value]=max_value
            res=np.vstack((res,np.copy(tmp))) if res is not None else np.copy(tmp)
        adata.X=res
    else:
        adata=sc.pp.scale(adata,zero_center=True, max_value=max_value,copy=True)
    return adata

    

                      

def OriginalClustering(adata,resolution=1.2,n_neighbors=20,n_comps=50,n_PC=20,n_job=4,dotsne=True,doumap=True,dolouvain=True):
    #Do PCA directly
    sc.tl.pca(adata,n_comps=n_comps)
    n_pcs=n_PC if n_PC<n_comps else n_comps
    #Do tsne based pca result
    if dotsne:
        sc.tl.tsne(adata,random_state=2,learning_rate=150,n_pcs=n_PC,n_jobs=n_job)
        #Save original X
        adata.obsm["X_tsne.ori"]=adata.obsm['X_tsne']
    #Do umap 
    if doumap:
        sc.pp.neighbors(adata,n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        #Save original
        adata.obsm['X_umap.ori']=adata.obsm['X_umap']
    if dolouvain:
        sc.tl.louvain(adata,resolution=resolution)
        adata.obs['louvain_ori']=adata.obs['louvain']
    print("OriginalClustering has completed!!!")

##
def first2prob(adata):
    first2ratio=[name for name in adata.uns.key() if str(name).startswith("prob_matrix")]
    for key_ in first2ratio:
        q_pred=adata.uns[key_]
        q_pred_sort=np.sort(q_pred,axis=1)
        y=q_pred_sort[:,-1]/q_pred_sort[:,-2]
        adata["first2ratio_"+str(key_).split("matrix")[1]]=y



#or 
def getdims(x=(10000,200)):
    """
    return the dims for network
    """
    assert len(x)==2
    n_sample=x[0]
    if n_sample>20000:# may be need complex network
        dims=[x[-1],128,32]
    elif n_sample>10000:#10000
        dims=[x[-1],64,32]
    elif n_sample>5000: #5000
        dims=[x[-1],32,16] #16
    elif n_sample>2000:
        dims=[x[-1],128]
    elif n_sample>500:
        dims=[x[-1],64]
    else:
        dims=[x[-1],16]
    #dims=[x[-1],64,32] if n_sample>10000 else [x[-1],32,16]
    return dims

