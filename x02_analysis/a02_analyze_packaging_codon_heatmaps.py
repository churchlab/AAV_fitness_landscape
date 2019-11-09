#!/usr/bin/env python
# coding: utf-8

# # Heatmaps for all codon positions across Rep and CMV libraries

# In[1]:


import os 
import sys 

import pandas as pd 
pd_idx =pd.IndexSlice
import numpy as np
import seaborn as sns 
from matplotlib import (gridspec,
                        ticker,
                        pyplot as plt)
from scipy.stats import variation
from scipy.stats import ks_2samp,mannwhitneyu,pearsonr

import common
from common import DESIRED_AA_COD_ORD
from common import plot_heatmap
sys.path.append('../x01_process_data/')
import x02_load_dataframes
import x03_compute_selections
sys.path.append('../settings/')
from paths_new import FIGURES_DIR
from paths_new import META_DIR
from paths_new import DESIRED_AA_ORD
from paths_new import WT_PLOTTING_DF
from paper_settings import PAPER_PRESET
from paper_settings import PAPER_FONTSIZE
from paper_settings import save_fig

PAPER_PRESET['font'] = 'monospace'
sns.set(**PAPER_PRESET)

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


packaging_counts = x02_load_dataframes.load_packaging_df()
packaging_counts.head()


# In[8]:


package_bio_codon_plotting_selection = x03_compute_selections.compute_packaging_selection(
    packaging_counts,level='codon', wt_norm=True, sum_measurments=True, for_plotting=True)
package_bio_codon_plotting_selection.head()


# ### CMV substituions

# In[9]:


for x in range(0,735, 147):
    plot_heatmap(package_bio_codon_plotting_selection,
                 lib_type='sub', 
                 promoter='CMV',
                 lib_num='0', 
                 range_in=(np.arange(x,x+147+1)),
                 fig_dimensions=[9,3],
                 dot_size=1,
                 text_size=0,
                 tick_size=3,
                 plot_white=False,
                 line_width = .5,
                 min_max = (-5,5),
                 return_df = False,
                 cmap='RdBu_r',
                 plt_cbar=False, 
                 save=True, 
                 save_name = None)


# ### CMV insertions

# In[10]:


for x in range(0,735, 147):
    plot_heatmap(package_bio_codon_plotting_selection,
                 lib_type='ins', 
                 promoter='CMV',
                 lib_num='0', 
                 range_in=(np.arange(x,x+147+1,.5)),
                 fig_dimensions=[9,3],
                 dot_size=1,
                 text_size=0,
                 tick_size=3,
                 plot_white=False,
                 line_width = .5,
                 min_max = (-5,5),
                 return_df = False,
                cmap='RdBu_r',
                plt_cbar=False, 
                 save=True, 
                save_name = None)


# ### Rep substituions 

# In[11]:


for x in range(0,735, 147):
    plot_heatmap(package_bio_codon_plotting_selection,
                 lib_type='sub', 
                 promoter='Rep',
                 lib_num='0', 
                 range_in=(np.arange(x,x+147+1)),
                 fig_dimensions=[9,2.5],
                 dot_size=1,
                 text_size=0,
                 tick_size=2.7,
                 plot_white=False,
                 line_width = .5,
                 min_max = (-5,5),
                 return_df = False,
                cmap='RdBu_r',
                plt_cbar=False, 
                 save=True, 
                save_name = None)


# ### Rep Insertions

# In[12]:


for x in range(0,735, 147):
    plot_heatmap(package_bio_codon_plotting_selection,
                 lib_type='ins', 
                 promoter='Rep',
                 lib_num='0', 
                 range_in=(np.arange(x,x+147+1,.5)),
                 fig_dimensions=[9,2.5],
                 dot_size=1,
                 text_size=0,
                 tick_size=2.7,
                 plot_white=False,
                 line_width = .5,
                 min_max = (-5,5),
                 return_df = False,
                cmap='RdBu_r',
                plt_cbar=False, 
                 save=True, 
                save_name = None)

