from setuptools import setup, find_packages

setup(name = 'desc',
			version = '1.0.0.5',
			description = 'Deep Embedded Single-cell RNA-seq Clustering',
			long_description = 'DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm constructs a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learning cluster-specific gene expression representation and cluster assignment based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances biological and technical differences between clusters, and reduces the influence of batch effect. DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, which facilitates the identification of cells clustered with high-confidence and interpretation of results.',
			classifiers = [
        'Development Status :: 3 - Alpha',
  	'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      	],
			url = 'https://github.com/eleozzr/desc',
			author = 'Xiangjie Li, Gang Hu, Mingyao Li, Yafei Lyu',
			author_email = 'ele717@163.com, huggs@nankai.edu.cn, mingyao@pennmedicine.upenn.edu, lyuyafei@gmail.com',
			license = 'MIT',
			packages = find_packages(),
			include_package_data=True,
			install_requires = [
				'matplotlib>=2.2'
				'pydot', 
				'tensorflow==1.7.0',
				'keras', 
				'scanpy',
				'louvain',
				'python-igraph',  
				'h5py',
				'pandas>=0.21', 
				],
			zip_safe = False
			)
			
