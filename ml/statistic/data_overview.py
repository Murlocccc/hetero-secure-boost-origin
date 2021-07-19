import types
from computing.d_table import DTable
from ml.utils.logger import LOGGER


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

def rubbish_clear(rubbish_list):
    """
    Temporary procession for resource recovery. This will be discarded in next version because of our new resource recovery plan
    Parameter
    ----------
    rubbish_list: list of DTable, each DTable in this will be destroy
    """
    for r in rubbish_list:
        try:
            if r is None:
                continue
            r.destroy()
        except Exception as e:
            LOGGER.warning("destroy Dtable error,:{}, but this can be ignored sometimes".format(e))