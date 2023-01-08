# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:01:04 2023

@author: asadm
"""
import numpy as np
import pandas as pd

def get_shares_and_indexes(data, share_row=None, index_column=None):
    """
    Gives dataframes of % against a row and ratio of % against a column
    
    Dataframe of shares is calculated by taking the % of all the rows against the specified row.
    Dataframe of indexes is calculated by taking the ratio of all the columns against the specified column of the Shares.
    
    Parameters
    ----------
    data : DataFrame
        Dataframe with numerical data.
    share_row : row index, optional
        The row to be considered as the denominator for the %.
        If None then last row is used.
    index_column : column index, optional
        The column to be considered as the denominator for the ratio.
        If None then last column is used.

    Returns
    -------
    shares : DataFrame
        A dataframe with % calculated against a row.
    indexes : DataFrame
        A dataframe with ratios calculated against a column.
        
    Examples
    -------
    >>> data
                     A   B
    itemCoffee      84  43
    itemPastry      17  11
    itemJuice        4   1
    itemSandwich     8   4
    itemColdDrink   38  23
    itemChips        1   0
    itemNone         5   2
    Base           122  65
    >>> shares, indexes = get_shares_and_indexes(data)
    >>> shares
                          A         B
    itemCoffee     0.688525  0.661538
    itemPastry     0.139344  0.169231
    itemJuice      0.032787  0.015385
    itemSandwich   0.065574  0.061538
    itemColdDrink  0.311475  0.353846
    itemChips      0.008197  0.000000
    itemNone       0.040984  0.030769
    Base           1.000000  1.000000
    >>> indexes
                          A    B
    itemCoffee     1.040793  1.0
    itemPastry     0.823398  1.0
    itemJuice      2.131148  1.0
    itemSandwich   1.065574  1.0
    itemColdDrink  0.880257  1.0
    itemChips           inf  NaN
    itemNone       1.331967  1.0
    Base           1.000000  1.0

    """
    #Validation checks
    if not isinstance(data, pd.core.frame.DataFrame):
        raise ValueError(f'data should be of type DataFrame instead of {str(type(data))}')
        
    if share_row is None:
        share_row = data.index[-1]
    if index_column is None:
        index_column = data.columns[-1]
        
    shares = data.divide(data.loc[share_row], axis=1)
    indexes = shares.copy()
    indexes = indexes.divide(indexes[index_column], axis=0)
    return shares, indexes

def single_select(
        data,
        column,
        weights=None
        ):
    """
    Profile a single select survey question.
    
    Generates the weighted response rate of a single-select question. It enables getting the response
    rate of the same question across different subsets of the dataframe in one go.
    
    Parameters
    ----------
    data : dict{str:DataFrame} or DataFrame
        Dataframes of survey data, where rows are respondents and columns are questions.
    column : column name or index
        Column name of the question to profile.
    weights : 1-D array-like or column name/index; optional
        Column name of the weights column or weights given separately in an array.
        If none is given then each row has weight=1.

    Returns
    -------   
    values : DataFrame
        A dataframe with weighted response rate of each dataframe in a separate column, with choices(unique values in column) in the index.
    
    Examples
    -------
    >>> data
         Gender     Occupation
    0    Female        Student
    1    Female        Student
    2      Male       Employed
    3    Female        Student
    4      Male        Student
    ..      ...            ...
    117    Male  Self-employed
    118    Male       Employed
    119    Male        Student
    120  Female       Employed
    121    Male       Employed
    >>> students = data[data["Occupation"]=="Student"]
    >>> students
     Gender     Occupation
    0    Female    Student
    1    Female    Student
    3    Female    Student
    4      Male    Student
    ..      ...        ...   
    112    Male    Student
    113  Female    Student
    115    Male    Student
    116    Male    Student
    119    Male    Student
    >>> single_select({"All":data, "Students":students}, "Gender")
            All  Students
    Female   65        24
    Male     57        18
    Base    122        42
    """
    #Validation checks
    if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, dict) and not all([isinstance(d, pd.core.frame.DataFrame) for _, d in data.items()]):
        raise ValueError(f'data should be of type DataFrame or dict {{str:DataFrame}} instead of {str(type(data))}')
    
    if isinstance(data, pd.core.frame.DataFrame):
        constructed_dict = {"Data":data}
    else:
        constructed_dict = {_:d for _, d in data.items()}

    if isinstance(weights,np.ndarray) or isinstance(weights, list) or weights is None:
        if weights is None:
            weights = 1
        weight_col = "_weight_dummy"
        constructed_dict = {_:d.assign(_weight_dummy=weights) for _, d in constructed_dict.items()}
    else:
        weight_col = weights

    for _, d in constructed_dict.items():
        if weight_col not in d.columns:
            raise Exception(f"Given weight column is not present in {_}")
        if column not in d.columns:
            raise Exception(f"Column is not present in {_}")

    #
    values = None
    for name, audience in constructed_dict.items():
        x = audience.groupby(column)[weight_col].sum().to_frame()
        x.columns = [name]
        if values is None:
            values = x
        else:
            values = values.join(x, how="outer")
    
    totals = values.sum().to_list()
    totals = pd.DataFrame([totals], columns=list(values.columns), index=["Base"])
    values = pd.concat([values, totals])
    return values

def multi_select(
        data,
        columns,
        weights=None,
        logical_one=1):
    """
    Profile a multi-select survey question.
    
    Generates the weighted response rate of a multi-select question. It enables getting the response
    rate of the same question across different subsets of the dataframe in one go. It expects the multi-select
    question to be across columns, i.e. each choice of the question in a separate column.

    Parameters
    ----------
    data : dict{str:DataFrame} or DataFrame
        Dataframes of survey data, where rows are respondents and columns are questions.
    column : list(column name or index)
        Column names of the question to profile.
    weights : 1-D array-like or column name/index; optional
        Column name of the weights column or weights given separately in an array.
        If none is given then each row has weight=1.
    logical_one : scalar, optional
        The value to be considered as "1",true or positive response. The default is 1.

    Returns
    -------
    values : DataFrame
        A dataframe with weighted response rate of each dataframe in a separate column, with choices(columns) in the index..
        
    Examples
    -------
    >>> data
         Gender  itemCoffee  itemPastry  itemColdDrink
    0    Female           1           0              0
    1    Female           0           1              1
    2      Male           1           0              0
    3    Female           1           0              0
    4      Male           1           0              0
    ..      ...         ...         ...            ...
    117    Male           1           0              0
    118    Male           1           1              1
    119    Male           1           0              1
    120  Female           1           0              0
    121    Male           1           0              0
    >>> females = data[data["Gender"]=="Female"]
         Gender  itemCoffee  itemPastry  itemColdDrink
    0    Female           1           0              0
    1    Female           0           1              1
    3    Female           1           0              0
    5    Female           0           0              1
    6    Female           1           0              0
    ..      ...         ...         ...            ...
    98   Female           1           0              0
    107  Female           0           0              1
    111  Female           0           0              1
    113  Female           0           0              1
    120  Female           1           0              0
    >>> multi_select({"All": data, "Females": females}, ["itemCoffee", "itemPastry", "itemColdDrink"])
                   All  Females
    itemCoffee      84       43
    itemPastry      17       11
    itemColdDrink   38       23
    Base           122       65
    """
    #Validation checks
    if not isinstance(data, pd.core.frame.DataFrame) and not isinstance(data, dict) and not all([isinstance(d, pd.core.frame.DataFrame) for _, d in data.items()]):
        raise ValueError(f'data should be of type DataFrame or dict {{str:DataFrame}} instead of {str(type(data))}')
    if not isinstance(columns, list):
        raise ValueError(f'column should be of type list(column_names) instead of {str(type(data))}')
    
    if isinstance(data, pd.core.frame.DataFrame):
        constructed_dict = {"Data":data}
    else:
        constructed_dict = {_:d for _, d in data.items()}

    if isinstance(weights,np.ndarray) or isinstance(weights, list) or weights is None:
        if weights is None:
            weights = 1
        weight_col = "_weight_dummy"
        constructed_dict = {_:d.assign(_weight_dummy=weights) for _, d in constructed_dict.items()} 
    else:
        weight_col = weights

    for _, d in constructed_dict.items():
        if weight_col not in d.columns:
            raise Exception(f"Given weight column is not present in {_}")
        if not set(columns).issubset(set(d.columns)):
            raise Exception(f"columns is not subset of {_}")
    #
    values = []
    index = []
    for col in columns:
        index.append(col)
        values.append([audience[audience[col]==logical_one][weight_col].sum() for name, audience in constructed_dict.items()])
    base_row = [audience[~audience[columns].isna().any(axis=1)][weight_col].sum() for name, audience in constructed_dict.items()]
    index.append("Base")
    values.append(base_row)
    values = pd.DataFrame(values, columns=list(constructed_dict.keys()), index=index)
    return values
    
    
    