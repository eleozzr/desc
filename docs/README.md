
## Deep Embedding for Single-cell Clustering (DESC)

DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm constructs a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learning cluster-specific gene expression representation and cluster assignment based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances biological and technical differences between clusters, and reduces the influence of batch effect. DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, which facilitates the identification of cells clustered with high-confidence and interpretation of results. 

![DESC workflow](assets/images/desc_workflow.png)

For thorough details, see our paper: [https://www.nature.com/articles/s41467-020-15851-3](https://www.nature.com/articles/s41467-020-15851-3)
<br>


## Usage

The [**desc**](https://github.com/eleozzr/desc) package is an implementation of deep embedding for single-cell clustering. With desc, you can:

- Preprocess single cell gene expression data from various formats.
- Build a low-dimensional representation of the single-cell gene expression data.
- Obtain soft-clustering assignments of cells.
- Visualize the cell clustering results and  the  gene expression patterns.

Because of the difference between `tensorflow 1*` and `tensorflow 2*`, we updated our [desc](https://github.com/eleozzr/desc) algorithm into two version such that it can be compatible with `tensorflow 1*` and `tensorflow 2*`, respectively. 

1. For  `tensorflow 1*`, we released `desc(2.0.3)`. Please see our jupyter notebook example [desc_2.0.3_paul.ipynb](./desc_2.0.3_paul.ipynb)
2. For  `tensorflow 2*`, we released `desc(2.1.1)`. Please see our jupyter notebook example [desc_2.1.1_paul.ipynb](./desc_2.1.1_paul.ipynb)


<br>

# Installation

To install  `desc` package you must make sure that your python version is either  `3.5.x` or `3.6.x`. If you don’t know the version of python you can check it by:
```python
>>>import platform
>>>platform.python_version()
#3.5.3
>>>import tensorflow as tf
>>> tf.__version__
#1.7.0
```
**Note:** Because desc depend on `tensorflow`, you should make sure the version of `tensorflow` is lower than `2.0` if you want to get the same results as the results in our paper.
Now you can install the current release of `desc` by the following three ways.

* PyPI  
Directly install the package from PyPI.

```bash
$ pip install desc
```
**Note**: you need to make sure that the `pip` is for python3，or we should install desc by
```bash 
python3 -m pip install desc 
#or
pip3 install desc
```

If you do not have permission (when you get a permission denied error), you should install desc by 

```bash
$ pip install --user desc
```

* Github  
Download the package from [Github](https://github.com/eleozzr/desc) and install it locally:

```bash
git clone https://github.com/eleozzr/desc
cd desc
pip install .
```

* Anaconda

If you do not have  Python3.5 or Python3.6 installed, consider installing Anaconda  (see [Installing Anaconda](https://docs.anaconda.com/anaconda/install/)). After installing Anaconda, you can create a new environment, for example, `DESC` (*you can change to any name you like*):

```bash
conda create -n DESC python=3.5.3
# activate your environment 
source activate DESC 
git clone https://github.com/eleozzr/desc
cd desc
python setup.py build
python setup.py install
# now you can check whether `desc` installed successfully!
```

Please check desc [Tutorial](https://eleozzr.github.io/desc/tutorial.html) for more details. And we also provide a [simple example](./paul_desc.md) for reproducing the results of Paul's data in our paper.


<br>

## Contributing

Souce code: [Github](https://github.com/eleozzr/desc)  

We are continuing adding new features. Bug reports or feature requests are welcome.

<br>


## References

Please consider citing the following reference:

- Xiangjie Li, Yafei Lyu, Jihwan Park, Jingxiao Zhang, Dwight Stambolian, Katalin Susztak, Gang Hu, Mingyao Li. Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis. 2019. bioRxiv 530378; doi: [https://doi.org/10.1101/530378](https://www.biorxiv.org/content/10.1101/530378v1?rss=1)
<br>
