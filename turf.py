# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:01:42 2023

@author: asadm
"""
import numpy as np
import pandas as pd
from itertools import combinations
import heapq as hq

def turf(
        data,
        columns,
        size,
        weights=None,
        min_response=None,
        forced_alt=None,
        mxclusive_alt=None,
        top=None
        ):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    columns : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.
    weights : TYPE, optional
        DESCRIPTION. The default is None.
    min_response : TYPE, optional
        DESCRIPTION. The default is None.
    forced_alt : TYPE, optional
        DESCRIPTION. The default is None.
    mxclusive_alt : TYPE, optional
        DESCRIPTION. The default is None.
    top : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #Validation checks
    if not isinstance(data, pd.core.frame.DataFrame):
        raise ValueError(f'data should be of type DataFrame or 2-D array-like convertible to DataFrame instead of {str(type(data))}')
    if not isinstance(columns, list):
        raise ValueError(f'columns should be of type list(column_names) instead of {str(type(data))}')
    if not isinstance(size, int):
        raise ValueError(f'size should be of type int instead of {str(type(data))}')
    if min_response is not None and not isinstance(min_response, int) and not isinstance(min_response, float):
        raise ValueError(f'min_response should be of type int or float instead of {str(type(data))}')
    if forced_alt is not None and not isinstance(forced_alt, list):
        raise ValueError(f'forced_alt should be of type list(column_names) instead of {str(type(data))}')
    if mxclusive_alt is not None and not isinstance(mxclusive_alt, list):
        raise ValueError(f'mxclusive_alt should be of type list(column_names) instead of {str(type(data))}')
    if top is not None and not isinstance(top, int):
        raise ValueError(f'top should be of type int instead of {str(type(data))}')

    if isinstance(weights,np.ndarray) or isinstance(weights, list) or weights is None:
        if weights is None:
            weights = 1
        weight_col = "_weight_dummy"
        constructed_data = data.assign(_weight_dummy=weights)
    else:
        constructed_data = data.copy()
        weight_col = weights

    if weight_col not in constructed_data.columns:
        raise Exception(f"Given weight column is not present in data")
    if not set(columns).issubset(set(constructed_data.columns)):
        raise Exception(f"columns is not subset of data")
    if forced_alt is not None and not set(forced_alt).issubset(set(constructed_data.columns)):
        raise Exception(f"forced_alt is not subset in data")
    if mxclusive_alt is not None and not set(mxclusive_alt).issubset(set(constructed_data.columns)):
        raise Exception(f"mxclusive_alt is not subset in data")
    if forced_alt is not None and mxclusive_alt is not None and not set(forced_alt).isdisjoint(set(mxclusive_alt)):
        raise Exception(f"mxclusive_alt should not have common elements in forced_alt")
    if size>len(columns):
        raise Exception(f"size should be lesser than the total number of columns")
    if forced_alt is not None and (len(forced_alt)>len(columns) or len(forced_alt)>size):
        raise Exception(f"Number of forced_alt should be lesser than size and the total number of columns")
    #
    constructed_data = constructed_data[columns+[weight_col]]
    filt_columns = [col for col in columns]
    updated_size = size
    if min_response is not None:
        filt_columns = [col for col in filt_columns if constructed_data[constructed_data[col]==1][weight_col].sum()>=min_response]
    if forced_alt is not None:
        filt_columns = [col for col in filt_columns if col not in forced_alt]
        updated_size -= len(forced_alt)
    
    turf = []
    combi_iter = combinations(filt_columns, updated_size)
    for combi in combi_iter:
        cols = list(combi)
        if forced_alt is not None:
            cols += forced_alt
        if mxclusive_alt is not None and len(set(cols).intersection(set(mxclusive_alt)))>1:
            continue
        reach_scores = constructed_data[cols].max(axis=1) * constructed_data[weight_col]
        freq_scores = constructed_data[cols].sum(axis=1) * constructed_data[weight_col]
        if top is not None:
            if len(turf)<top:
                hq.heappush(turf, [reach_scores.sum(), freq_scores.sum(), ", ".join(cols)])
            else:
                hq.heappushpop(turf, [reach_scores.sum(), freq_scores.sum(), ", ".join(cols)])
        else:
            turf.append([reach_scores.sum(), freq_scores.sum(), ", ".join(cols)])
    turf = pd.DataFrame(turf, columns=["Reach", "Frequency", "Combination"])
    return turf.sort_values(by=["Reach", "Frequency"], ascending=False, ignore_index=True)
    
    