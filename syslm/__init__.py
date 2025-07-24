# syslm package
import pkg_resources
import pandas as pd

def load_example(name):
    """
    Load example data by name.
    name: one of 'otu_phylum_CLR.csv', 'dynamic_weights.h5', etc.
    """
    path = pkg_resources.resource_filename("syslm", f"example_data/{name}")
    return path  # 可用于 pd.read_csv(path) 或 h5py.File(path)