#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from matplotlib import rcParams
rcParams['font.family'] = 'Lato' # if you want Arial font you need to install first 
from paths_new import FIGURES_DIR
import matplotlib.pyplot as plt
PAPER_PRESET = {"style": "ticks", "font": "Lato", "context": "paper", 
                "rc": {"font.size":7,"axes.titlesize":7,
                       "axes.labelsize":7, 'axes.linewidth':0.5,
                       "legend.fontsize":6, "xtick.labelsize":6,
                       "ytick.labelsize":6, "xtick.major.size": 3.0,
                       "ytick.major.size": 3.0, "axes.edgecolor": "black",
                       "xtick.major.pad": 3.0, "ytick.major.pad": 3.0}}
PAPER_FONTSIZE = 7

def save_fig(fig,fig_name,transparent=True):
    fig_path = os.path.join(FIGURES_DIR, fig_name)
    fig.savefig(fig_path, bbox_inches='tight', dpi='figure',transparent=transparent)

