import types
from computing.d_table import DTable


def get_features_shape(data_instances: DTable):
    return len(data_instances.schema)

def is_empty_feature(data_instances: DTable):
    # TODO
    ...

def get_header(data_instances: DTable) -> list: 
    header = data_instances.schema.get('header') # ['x1', 'x2', 'x3' ... ]
    return header

def is_sparse_data(data_instances: DTable) -> bool:
    first_data = data_instances.first()
    if type(first_data).__name__ in ['ndarray', 'list']:
        return False

    data_feature = first_data.features
    if type(data_feature).__name__ == "ndarray":
        return False
    else:
        return True