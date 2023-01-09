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
    Calculates the total unduplicated reach and frequency of a column-combination.
    
    It is useful for getting the unduplicated reach and frequency of of combinations of length=size
    from survey questions dealing with user preferences. 
    TURF is used for optimizing portfolios where n=number of columns/products (each column containing
    the preference of a user liking that product or not, i.e. 1 and 0) and size=length of combinations
    to test the reach and frequency for, i.e. nCr.

    Parameters
    ----------
    data : DataFrame
        Dataframe of survey data, where rows are respondents and columns are questions.
    columns : list(column name or index)
        Column names of the question to profile. Columns should have 1s and 0s.
    size : int
        Length of the combinations to test for TURF.
    weights : 1-D array-like or column name/index; optional
        Column name of the weights column or weights given separately in an array.
        If none is given then each row has weight=1.
    min_response : int or float, optional
        Minimum weighted positive response rate of a column out of "columns" required for the column
        to be considered for TURF. The default is None.
    forced_alt : list(column name or index), optional
        Set of columns out of "columns" which are to be forced-added in every combination.
        It enables checking for reach of extending an exisiting portfolio with new products.
        The default is None.
    mxclusive_alt : list(column name or index), optional
        Set of columns out of "columns" which should be considered mutually exclusive.
        This will exclude the combinations which contains more than 1 column out of the given list.
        The default is None.
    top : int, optional
        The number of top n combinations based on reach that should be present in the output.
        It employs the use of a heap-queue to keep track of the top combinations.
        Should be used in case where value of nCr is expected to be too large. 
        The default is None.

    Returns
    -------
    DataFrame
        A dataframe with weighted Reach and Frequency for the combinations of length="size".
        
    Examples
    -------
    >>> data
         itemCoffee  itemPastry  itemJuice  ...  itemColdDrink  itemChips  itemNone
    0             1           0          0  ...              0          0         0
    1             0           1          0  ...              1          0         0
    2             1           0          0  ...              0          0         0
    3             1           0          0  ...              0          0         0
    4             1           0          0  ...              0          0         0
    ..          ...         ...        ...  ...            ...        ...       ...
    117           1           0          0  ...              0          0         0
    118           1           1          1  ...              1          0         0
    119           1           0          0  ...              1          0         0
    120           1           0          0  ...              0          0         0
    121           1           0          0  ...              0          0         0
    
    [122 rows x 7 columns]
    >>> turf(data, multi_cols, 3, top=10)
        Reach  Frequency                              Combination
     0    118        127      itemCoffee, itemColdDrink, itemNone
     1    117        139    itemCoffee, itemPastry, itemColdDrink
     2    115        123     itemCoffee, itemColdDrink, itemChips
     3    114        130  itemCoffee, itemSandwich, itemColdDrink
     4    114        126     itemCoffee, itemJuice, itemColdDrink
     5     96        106         itemCoffee, itemPastry, itemNone
     6     92        102        itemCoffee, itemPastry, itemChips
     7     91        109     itemCoffee, itemPastry, itemSandwich
     8     91        105        itemCoffee, itemPastry, itemJuice
     9     91         93          itemCoffee, itemJuice, itemNone
    >>> turf(data, multi_cols, 3, top=10, forced_alt=["itemCoffee"], 
             mxclusive_alt=["itemColdDrink", "itemPastry"])
        Reach  Frequency                              Combination
     0    118        127      itemColdDrink, itemNone, itemCoffee
     1    115        123     itemColdDrink, itemChips, itemCoffee
     2    114        130  itemSandwich, itemColdDrink, itemCoffee
     3    114        126     itemJuice, itemColdDrink, itemCoffee
     4     96        106         itemPastry, itemNone, itemCoffee
     5     92        102        itemPastry, itemChips, itemCoffee
     6     91        109     itemPastry, itemSandwich, itemCoffee
     7     91        105        itemPastry, itemJuice, itemCoffee
     8     91         93          itemJuice, itemNone, itemCoffee
     9     90         97       itemSandwich, itemNone, itemCoffee
    ^every combination above contains "itemCoffee" and none of the contains "itemColdDrink" 
    and "itemPastry" together.
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
    
    