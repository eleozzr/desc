
## Deep Embedding for Single-cell Clustering 

DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm constructs a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learning the cluster-specific gene expression representation and member assignments based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances the biological and the technical differences between clusters, and reduces the influence of the batch effect. DESC also enables soft clustering by assigning cluster-specific probability to each cell, which facilitates the identification of cells clusters with high-confidence and interpretation of results. 

![DESC workflow](docs/assets/images/desc_workflow.png)

For thorough details, see the preprint: [Bioxiv](https://www.biorxiv.org)
<br>

## Usage

The [**desc**](https://github.com/eleozzr/desc) package is an implementation of deep embedding for single-cell clustering. With desc, you can:

- Pre-process single cell expression data from varies of formats
- Build a low-dimensional representation of the single-cell expression data
- Obtain soft-clustering assignments of cells
- Visualize the cell clusters separation and the expression pattern
<br>

# Installation

To install the current release of `desc` you can choose:

* PyPI  
Directly install the package from PyPI:

```
$ pip install desc
```

If you do not have sudo rights (when you get a permission denied error): 

```
$ pip install --user desc
```

* Github  
Download the package from the Github and install it locally:

```
git clone https://github.com/eleozzr/desc
cd desc
pip install .
```

* Anaconda

If you do not have a working Python3.5 or Python3.6 installation, consider installing Anaconda , (see Installing Anaconda ). After installing Anaconda, you can create a new environment, for example, DESC:

```
conda create -n DESC python=3.5.3
# activate your environment 
source activate DESC 
git clone https://github.com/eleozzr/desc
cd desc
python setup.py build
python setup.py install
# now you can check whether `desc` installed successfully!
```

Please check DESC [Tutorial](https://eleozzr.github.io/desc/tutorial.html) for more detials.
<br>

## Contributing

Souce code: [Github](https://github.com/eleozzr/desc)  

We are continuing to add new features. Any kind of contribution, like bug reports or feature requests, are welcome.
<br>

## References

Please consider citing the references in case they are used.  
[1] DESC    
[2] DESC
<br>

## Licence

Copyright (C) 2019 Mingyao's Lab