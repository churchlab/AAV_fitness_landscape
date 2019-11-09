#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import sys

import pandas as pd 
pd_idx = pd.IndexSlice

sys.path.append('../settings/')


# In[2]:


def load_packaging_df(path='../data/dataframes/aav_packaging_all.csv.gz'):
    df = pd.read_csv(path,index_col=list(range(0,11)), header=[0,1,2,3,4,5])
    return df


# In[3]:


def load_antibody_df(path = '../data/dataframes/aav_antibody_all.csv.gz'):
    return pd.read_csv(path, index_col =list(range(0,11)))


# In[4]:


def load_thermostability_df(path='../data/dataframes/aav_thermostability_all.csv.gz'):
    return pd.read_csv(path, header=[0,1,2], index_col=list(range(0,11)))


# In[5]:


def load_mouse_df(path='../data/dataframes/aav_mouse_all.csv.gz'):
    return pd.read_csv(path,header=[0,1,2,3,4,5], index_col=list(range(0,11)))


# In[6]:


def load_MAAP_df(path='../data/dataframes/aav_maap_all.csv.gz'):
    df= pd.read_csv(path,index_col=list(range(0,11)), header=[0,1,2,3])
    return df


# In[ ]:




