#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


def codon_to_aa_selector(subset_df, df):
    '''
    Args: subset_df - a df with some subset of interesting alleles
          df - another df that yuou wish to grab the same subset of interesting alleles 
    Returns: A df with the same alleles from subset df, but the measurments for df 
    
    NOTE: This is most useful when one df is in codon format and another is in aa format, or vice versa 
    '''
    
    keep_indices = list(df.index.names)
    merge_indices =  [x for x  in list(subset_df.index.names[:len(keep_indices)]) if 'wt' not in x]
    print (merge_indices)
    return df.reset_index().merge(
        subset_df.reset_index()[merge_indices], on=merge_indices).set_index(keep_indices).drop_duplicates()

