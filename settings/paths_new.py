#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd

THIS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# Directories
DATA_DIR = os.path.join(THIS_MODULE_PATH, '../data/')
FASTQ_DIR = os.path.join(DATA_DIR, 'fastq')
COUNTS_DIR = os.path.join(DATA_DIR, 'counts')
DATAFRAMES_DIR = os.path.join(DATA_DIR, 'dataframes')
SAMPLE_SHEETS_DIR = os.path.join(DATA_DIR, 'sample_sheets')
META_DIR = os.path.join(DATA_DIR, 'meta')
FIGURES_DIR = os.path.join(DATA_DIR, 'figures')
LOOKUP_DF = pd.read_csv('../data/meta/AAV2scan_chip_lookup_table.txt')
MOUSE_VAL_DIR = os.path.join(DATA_DIR, 'mouse_validation_data')

WT_PLOTTING_DF = pd.read_csv(os.path.join(DATA_DIR,'meta','lookup_rc.txt'))
AA_INDEX_DF = pd.read_csv('../data/meta/aa_index_0_center_norm_df.csv')

DESIRED_AA_ORD = ["-","I", "L", "V", "A", "G", "M", "F", "Y", "W", "E", 
                    "D", "Q", "N", "H", "C", "R", "K", "S", "T", "P", "*"]


# In[ ]:




