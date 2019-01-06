# Copyright 2018 Xiangjie Li et al.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os,sys,argparse,textwrap
import scanpy.api as sc
#if __name__!="__main__":
#    try:
#except:
#import preprocessing as io
#from train import train

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Embeded Single Cell Clustering!!')
    parser.add_argument('-i','--input',type=str,help=textwrap.dedent(''' Input is raw count data in TSV(tsv,tab,data)/CSV or .h5ad(anndata) format'
                         'Row/col names are mandatory. Note that TSV/CSV files must be in gene×cell layout where rows are genes and
                          cols are cells(scRNA-seq convention). Use the -t/--transpose option if your count matrix in cell×gene layou                          t.H5AD files must be in cell*gene format(stats and scanpy convention), or '
                          'if directory, "matrix.mtx","barcodes.tsv","genes.tsv" must be included and separate by "tab separator". 
                           matrix.mtx example: 
                            %%MatrixMarket matrix coordinate integer general
                            35635 29065 17570845
                            63 1 1
                            82 1 1
                            114 1 1
                            152 1 1
                            167 1 3
                            .....

                            barcodes.tsv example:
                            cellname    batchID nUMI ...
                            AAACATACAACCAC  tenx    20000   ....
                            AAACATTGAGCTAC  tenx    10000   ....
                            AAACATTGATCAGC  tenx    12000
                            AAACATTGATCAGC  smartseq    300000  ....
                            or when no columns name the first columns will be regarded as  cellname
                            .....
                    
                            genes.tsv example:
                            genename    geneID
                            MIR1302-10  ensembl1
                            FAM138A ensembl1
                            OR4F5   ensembl1
                            RP11-34P13.7    ensembl1
                            RP11-34P13.8    ensembl1
                            or when no columns name the first columns will be regarded as  genename, if only have two columns, 
                            the second column will be regarded as gene name" '''))
    parser.add_argument('-o','--output',type=str,default='adata.h5ad',help="The file name of the output`anndata`")
    #transpose if layout is gene*cell
    parser.add_argument('-t','--transpose',type=bool,default=False,help="Transpose input matrix(default:False) only True when input is cell*gene")
    #parameter to pre filter cells
    parser.add_argument('--prefilter_cells',type=bool,default=True,help='Prefilter cells')
    parser.add_argument('--c_min_genes',type=int,default=1,help='')    
    parser.add_argument('--c_max_genes',type=int,default=None,help='')    
    parser.add_argument('--c_min_counts',type=int,default=None,help='')    
    parser.add_argument('--c_max_counts',type=int,default=None,help='')    
    #parameter to filter cells according to mito_percent
    parser.add_argument('--filter_cells_mito',type=bool,default=True,help='Whether filter cells according to mito_percent') 
    parser.add_argument('--mito_percent_cutoff',type=float, default=0.10,help='The cell with mito_percent>0.20 will be filtered.')
    parser.add_argument('-nUMIbefore_filter_MT_gene',type=bool,default=True,help='Compute n_counts and n_gene before filtering MT-ge                        ne if nUMIbefore_filter_MT_gene==True')
    #parameter to normalization cell

    parser.add_argument('--normalize_per_cell',type=bool,default=False,help='Normalize cells. if normalize_cell=False')
    parser.add_argument('--counts_per_cell_after',default=1e4,action='store_true',help=textwrap.dedent(''' `float` or `None`, optional (default: 
            `1e4`)If `None`, after normalization, each cell has a total count equal  to the median of the 
             *counts_per_cell* before normalization.'''))
    parser.add_argument('--counts_per_cell',default=None,action='store_true',help='`float` or `None`, optional(default:`None`) Precomputed counts per cell.')
    #parameter to pre filter genes
    parser.add_argument('--prefilter_genes',type=bool,default=True,help='Prefilter cells')
    parser.add_argument('--g_min_cells',type=int,default=1,help='')    
    parser.add_argument('--g_max_cells',type=int,default=None,help='')    
    parser.add_argument('--g_min_counts',type=int,default=None,help='')    
    parser.add_argument('--g_max_counts',type=int,default=None,help='')  
    #parameter to find variable genes
    parser.add_argument('--find_variable_genes',type=bool,default=True,help='find variable genes, the parameter is same as scanpy.api.pp.filter_gene_dispersion')
    parser.add_argument('--flavor',type=str,default='seurat',help=textwrap.dedent(''' "{'seurat', 'cell_ranger'}, optional (default: 'seurat')
        Choose the flavor for computing normalized dispersion. If choosing
        'seurat', this expects non-logarithmized data - the logarithm of mean
        and dispersion is taken internally when `log` is at its default value
        `True`. For 'cell_ranger', this is usually called for logarithmized data
        - in this case you should set `log` to `False`. In their default
        workflows, Seurat passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.min_mean=0.0125, max_mean=3, min_disp=0.5, max_disp=`None` : `float`, optional
        If `n_top_genes` unequals `None`, these cutoffs for the means and the
        normalized dispersions are ignored." '''))
    parser.add_argument('--min_mean',type=float,default=0.0125)
    parser.add_argument('--max_mean',type=float,default=None)
    parser.add_argument('--min_disp',type=float,default=None)
    parser.add_argument('--max_disp',type=float,default=None)
    parser.add_argument('--n_top_genes',type=int,default=1000,help="Number of highly-variable genes to keep.")
    #parameter take lop1p
    parser.add_argument('--log1p',type=bool,default=True,help='Take log1p transformation after find variable genes')
    #parameter scale_by_group
    parser.add_argument('--scale_data',type=int,default=1,help='one of [0,1,2],0:no scale, 1:scale overall cells 2:scale by group')
    parser.add_argument('--group',type=str,default=None,help='Scale gene by group, group must be the columns of adata.')
    parser.add_argument('--max_value',type=str,default=None,help='The maxmum value after scale')
    
    #parameter desc model 
    parser.add_argument('--dims',type=int,default=[64,32],nargs='+',help="The number of nodes for each layer in encoder. if dims==-1, it will choose the number of nodes and layers automatically!!!")
    parser.add_argument('--alpha',type=float,default=1.0,help='the degree of t-distribution')
    parser.add_argument('--tol',type=float,default=0.005,help='Stop criterion, clustering procedure will be stoped when the difference ratio betwen the current iteration and last iteration larger than tol')
    parser.add_argument('--init',type=str,default='glorot_uniform',help='Initializing distribution,detail refer to keras')
    parser.add_argument('--n_clusters',type=int,default=None,help="The number of clusters for K-means methods, if not sepcified, it will use louvain method to initialize cluster centroid")
    parser.add_argument('--louvain_resolution',type=float,default=[0.4],nargs='+',help="resolution for louvian method")
    parser.add_argument('--n_neighbors',type=int,default=15,help="The neighbors for connectivity matrix of cell")
    parser.add_argument('--pretrain_epochs',type=int,default=300,help='the number of epochs for autoencoder model')
    parser.add_argument('--batch_size',type=int,default=256,help='the batch size for autoencoder model and clustering model')
    parser.add_argument('--activation',type=str,default='relu',help="the activation function for autoencoder model,which can be `elu,selu,softplus,tanh,siogmid et al.`, for detail please refer to`https://keras.io/activations/")
    parser.add_argument('--actincenter',type=str,default='tanh',help=' the activation function for the last layer in encoder and decoder model')
    parser.add_argument('--drop_rate_SAE',type=float,default=0.2,help='The drop rate for Stacked autoencoder, which just for  finetuning')
    parser.add_argument('--is_stacked',type=bool,default=True,help='The model will be pretrained by stacked autoencoder if is_stacked==True')
    parser.add_argument('--use_earlyStop',type=bool,default=True,help='Stops training if loss does not improve if given min_delta=1e-4, patience=10.')
    parser.add_argument('--use_ae_weights',type=bool,default=True,help='if True and ae_weights.h5 savedi in /`save_dir`/ae_weights.h5, it will use use_ae_weights')
    parser.add_argument('--save_encoder_weights',type=bool,default=False,help='if True save ae_weight`epoch`.h5')
    parser.add_argument('--save_dir',type=str,default='result_tmp',help='The  directory for result')
    parser.add_argument('--max_iter',type=int,default=1000,help=' The maximum iteration for clustering')
    parser.add_argument('--epochs_fit',type=int,default=4,help='updateing clustering probability for every epochs_fit. If sample size is samll, we tend to set a larger value')
    parser.add_argument('--num_Cores',type=int,default=20,help='How many cpus use during tranning. if num_Cores > the max cpus in our computer , num_Cores will use  a half of cpus in your computer.')
    parser.add_argument('--use_GPU',type=bool,default=False,help='It will use GPU to train model if GPU is avaliable')
    parser.add_argument('--random_seed',type=int,default=201809,help='he random seed for random.seed,,,numpy.random.seed,tensorflow.set_random_seed')
    parser.add_argument('--verbose',type=bool,default=False,help='It will ouput the model summary if verbose==True.')
    parser.add_argument('--do_tsne',type=bool,default=True,help='It will do tsne coordinate if do_tsne==True')
    return parser.parse_args()
 
def main():
    args=parse_args()
    print(args)
    from . import preprocessing as io
    from .train import train
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError('DESC requires tensorflow. please follow instructions at https://www.tensorflow.org/install/ to install it')  
    #Step1. create data
    adata=io.read_dataset(args.input,transpose=args.transpose) 
    #Step2. prefilter data
    if args.prefilter_cells: 
        io.prefilter_cells(adata,min_counts=args.c_min_genes,max_counts=args.c_max_counts,min_genes=args.c_min_genes,max_genes=args.c_max_genes)
    #filter cells according mito_percent and filter MT_gene, and computer nUMI nGene before and after filter MT_genes
    if args.filter_cells_mito:
        io.filter_cells_mito(adata,mito_percent_cutoff=args.mito_percent_cutoff,nUMIbefore_filter_MT_gene=args.nUMIbefore_filter_MT_gene) 
    #normalize data
    if args.normalize_per_cell:
        sc.pp.normalize_per_cell(adata,counts_per_cell_after=args.counts_per_cell_after,counts_per_cell=args.counts_per_cell)
        adata.raw=sc.pp.log1p(adata,copy=True)

    ## prefilter genes
    if args.prefilter_genes: 
        io.prefilter_genes(adata,min_counts=args.g_min_counts,max_counts=args.g_min_counts,min_cells=args.g_min_cells,max_cells=args.g_max_cells)
    #find variable genes
    if args.find_variable_genes: 
        assert args.flavor in ["seurat","cell_ranger"],"flavor must be one of{'seurat', 'cell_ranger'}"
        sc.pp.filter_genes_dispersion(adata,max_mean=args.max_mean,min_disp=args.min_disp,max_disp=args.max_disp,n_top_genes=args.n_top_genes)
    #take log-transformation
    if args.log1p:
        sc.pp.log1p(adata)
    if args.scale_data:
        assert args.scale_data in [0,1,2],'`scale_data` must be of one of 0,1,2'
        adata=io.scale_by_group(adata,groupby=args.group,max_value=args.max_value)
    #from . import train
    dims=[adata.shape[-1]]+args.dims if args.dims[0]!=-1 else io.getdims(adata.shape)
    adata=train(adata,
            alpha=args.alpha,
            tol=args.tol,
            init=args.init,
            n_clusters=args.n_clusters,
            louvain_resolution=args.louvain_resolution,
            n_neighbors=args.n_neighbors,
            pretrain_epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            activation=args.activation,
            actincenter=args.actincenter,
            drop_rate_SAE=args.drop_rate_SAE,
            is_stacked=args.is_stacked,
            use_earlyStop=args.use_earlyStop,
            use_ae_weights=args.use_ae_weights,
	    save_encoder_weights=save_encoder_weights,
            save_dir=args.save_dir,
            max_iter=args.max_iter,
            epochs_fit=args.epochs_fit,
            num_Cores=args.num_Cores,
            use_GPU=args.use_GPU,
            random_seed=args.random_seed,
            verbose=args.verbose,
	    do_tsne=args.do_tsne)
    print(adata)
    adata.write(os.path.join(args.save_dir,"adata_desc.h5ad"))
# no test

