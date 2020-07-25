#from . import original as og
from . import tools
from . import models
from . import datasets

from anndata import read_h5ad
from scanpy.preprocessing import normalize_per_cell,normalize_total,highly_variable_genes, log1p, scale

from .models.desc import train
from .tools.test import run_desc_test
from .tools.read import read_10X
from .tools.write import write_desc_result
from .tools.preprocessing import scale_bygroup
#from .tools.downstream import run_tsne
__version__ = '2.1.1'



