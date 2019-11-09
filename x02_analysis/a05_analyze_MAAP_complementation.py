#!/usr/bin/env python
# coding: utf-8

# # MAAP complementation analysis 
# 
# Here we analyze the data supporting discovery of the new gene MAAP.  
# Analysis includes  
# - constructuing a frameshift lookup table for detemrining effects 
# - global look at MAAPs affect across gene, with or without trans complementation 
# - bootstrap method for constrcuting p-values signfying stop codon importance in frame 2 
# - valiation data for individual mutants

# In[1]:


import os
import sys 

import pandas as pd 
pd_idx =pd.IndexSlice
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats
import scipy.special as special

import common 

sys.path.append('../x01_process_data/')
import x02_load_dataframes
import x03_compute_selections

sys.path.append('../settings/')
from paths_new import FIGURES_DIR
from paths_new import META_DIR
from paths_new import DESIRED_AA_ORD
from paths_new import LOOKUP_DF
from paths_new import DATA_DIR
from paper_settings import PAPER_PRESET
from paper_settings import PAPER_FONTSIZE
from paper_settings import save_fig
sns.set(**PAPER_PRESET)

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### load in data, need packaging data from MAAP experiments as well as original packaging data 

# In[2]:


maap_counts = x02_load_dataframes.load_MAAP_df()
package_counts = x02_load_dataframes.load_packaging_df()


# ### pEL plasmid labels describe the complement experiment which was run during packaging 
# - pRep: only inlcude the pRep plasmid to facilitate normal packaging in library 
# - pEK_254: include a plasmid which contains pRep as well as the downstream wildtype MAAP region 
# - pEK_256: include a plasmid which contains the downstream MAAP region, but with a stop codon substitution 
# - pEK_258: same as above, but with a mutation to MAAP start codon   
# we also ran bioligically packaging experiment replicates as well 

# In[3]:


maap_counts.head()


# ### compute selection from the counts

# In[4]:


maap_selection = x03_compute_selections.compute_maap_selection(package_counts, maap_counts)


# In[5]:


maap_selection.head()


# ### constrcut a table which informs what mutations were made in frame 2 
# since the library contains all single mutations in frame 1, we need to convert these to the possible mutations made in frame 2 

# In[6]:


frameshift_meta_lookup_df = common.make_all_frame_lookup(LOOKUP_DF)


# merge this new table with the selection data, and construct a few more useful columns for analysis

# In[7]:


maap_selection_and_lookup = frameshift_meta_lookup_df.merge(maap_selection.reset_index(), on=[ 'abs_pos','aa', 'codon'], how='inner')

# add easy stop identifier 
stop_obj = []
for idx, row in maap_selection_and_lookup.iterrows():
    if row['maap_aa_swap_first'] =='*' or row['maap_aa_swap_second'] =='*':
        x =1 
    else: 
        x=0
    stop_obj.append(x)
maap_selection_and_lookup['stop_maap'] = stop_obj

## set up useful indeces
maap_selection_and_lookup = maap_selection_and_lookup.set_index(list(maap_selection_and_lookup.columns[:31]))
maap_selection_and_lookup.set_index('stop_maap',append=True,inplace=True)
maap_selection_and_lookup.index = maap_selection_and_lookup.index.droplevel([-5,-4,-3,-2])
maap_selection_and_lookup.columns = pd.MultiIndex.from_tuples(maap_selection_and_lookup.columns)
maap_selection_and_lookup.drop_duplicates(inplace=True)

#an example with a stop codon
maap_selection_and_lookup.xs((50,'L'), level=['abs_pos', 'aa'],drop_level=False)


# ### from this, we take only the positions which contain at least one stop codon in frame 2 

# In[8]:


aa_pos_with_stops = maap_selection_and_lookup.xs(1, level='stop_maap',drop_level=False)


# In[9]:


aa_pos_with_stops.head()


# reshape the dataframe for plotting  

# In[10]:


aa_pos_with_stops_index = aa_pos_with_stops.reset_index(
        )[['aa','abs_pos']]

maap_idx_names = maap_selection_and_lookup.index.names
maap_aa_has_stop = aa_pos_with_stops_index.merge(maap_selection_and_lookup.reset_index(),on=['aa','abs_pos'],how='left').set_index(maap_idx_names)
maap_aa_has_stop_subset = pd.concat([maap_aa_has_stop.xs(['pEK_254','1'], level=[0,3],axis=1,drop_level=False),
                                     maap_aa_has_stop.xs(['pRep','2'], level=[0,3],axis=1,drop_level=False),
                                     maap_aa_has_stop.xs(['pEK_256','2'], level=[0,3],axis=1,drop_level=False),
                                     maap_aa_has_stop.xs(['pEK_258','2'], level=[0,3],axis=1,drop_level=False)],axis=1)
new_idx = maap_aa_has_stop_subset.reset_index(
    )[['abs_pos', 'aa', 'codon', 'stop_maap']].set_index(['abs_pos', 'aa', 'codon', 'stop_maap'])
maap_aa_has_stop_subset.index = new_idx.index
maap_aa_has_stop_subset.head()


# In[11]:


maap_aa_has_stop_subset_copy = maap_aa_has_stop_subset.copy()
maap_aa_has_stop_subset_copy.columns = maap_aa_has_stop_subset_copy.columns.droplevel([1,2,3])
maap_aa_has_stop_subset_copy.head()


# for each position compute the mean selection value for stop and non stop mutations

# In[12]:


maap_pos_mean_stop_unstack = maap_aa_has_stop_subset_copy.groupby(
    level=['abs_pos', 'aa', 'stop_maap']).mean().unstack(-1).dropna()


##  known position which makes stops 
maap_pos_mean_stop_unstack.query("abs_pos==53")


# compute the global difference between stops and non stops in MAAP codons

# In[13]:


maap_df = maap_pos_mean_stop_unstack.mean().unstack()
maap_df['div'] = maap_df[1] / maap_df[0]
maap_df['div_log2'] = maap_df['div'].apply(np.log2)
maap_df


# In[14]:


maap_pos_mean_stop_unstack_stop_diff = maap_pos_mean_stop_unstack.apply(np.log2).groupby(
    level=0, axis=1).diff(axis=1).dropna(axis=1).apply(lambda x: 2**x)
maap_pos_mean_stop_unstack_stop_diff.columns = maap_pos_mean_stop_unstack_stop_diff.columns.droplevel(1)
maap_pos_mean_stop_unstack_stop_diff.head()


# In[15]:


## only consider subs here becuase inseretions have general negatvie trend 
maap_pos_only_diff = maap_pos_mean_stop_unstack_stop_diff.loc[np.arange(0,735)].query('(abs_pos > 27) & (abs_pos<147)')


# In[16]:


maap_pos_only_diff.shape


# In[17]:


maap_pos_only_diff_outliers_rm = maap_pos_only_diff[(np.abs(stats.zscore(maap_pos_only_diff)) < 3).all(axis=1)]
maap_pos_only_diff_outliers_rm.shape


# drop two posiitons with strongest outlier data

# In[18]:


maap_pos_only_diff_drop = maap_pos_only_diff.drop([(50,'L'),(104,'V')])


# ### p-values comparing WT to complementation with MAAP (pEK 254) or MAAP with start (pEK_256) or stop (pEK_258) mutations 

# In[19]:


stats.mannwhitneyu(maap_pos_only_diff_drop['pRep'], maap_pos_only_diff_drop['pEK_254'])


# In[20]:


stats.mannwhitneyu(maap_pos_only_diff_drop['pRep'], maap_pos_only_diff_drop['pEK_256'])


# In[21]:


stats.mannwhitneyu(maap_pos_only_diff_drop['pRep'], maap_pos_only_diff_drop['pEK_258'])


# ### load in the individual mutant validation data (from qPCR)

# In[22]:


MAAP_dir = os.path.join(DATA_DIR, 'MAAP_validation_data')
individual_mut_validation_df = pd.read_csv(os.path.join(MAAP_dir,'20190604_individual_mutant_titer_with_meta.csv' ))
competition_mut_validation_df = pd.read_csv(os.path.join(MAAP_dir,'20190604_competition_mutant_titer_with_meta.csv' ))

competition_mut_validation_df.head()


# In[23]:


complement_dict = {'pRep':'+pRep', 'pRep-MAAP':'+pRep\nMAAP'}
individual_mut_validation_df['complement_plot'] =individual_mut_validation_df['complement'].    apply(lambda x: complement_dict[x])
individual_mut_validation_df.head()


# In[24]:


competition_mut_validation_df['complement_plot'] =competition_mut_validation_df['complement'].    apply(lambda x: complement_dict[x])
competition_mut_validation_df.head()


# In[25]:


def mapper(x):
    return np.log2(np.mean(x))


# ### plot affect of library with and without complementation, as well as individual mutants with or without MAAP in trans

# In[26]:


fig, ax =  plt.subplots(nrows=3, figsize = [2,3.5], gridspec_kw={'hspace':.6})

sns.barplot(x='level_2', y=0,order=['pRep','pEK_254', 'pEK_258','pEK_256' ], 
            data=maap_pos_only_diff_drop.stack().reset_index(),estimator=mapper,color='slategrey',errwidth=1, ax=ax[0])


ax[0].set_xticklabels(['+pRep','+pRep\nMAAP', u'+pRep\nMAAP\n27∆Start', u'+pRep\nMAAP\n32∆Stop'])
ax[0].set_ylabel(u'log2(∆Stop)')

ax[0].get_yaxis().set_ticks([0,-1])
ax[0].set_xlabel('')
ax[0].yaxis.set_label_coords(-.13, 0.5)


sns.barplot(x='Name', y='vector_genomes', hue='complement_plot',palette=sns.color_palette("Set2"), order = ["WT", "MAAP-27", "MAAP-59", "AAP-Stop"],
                data=individual_mut_validation_df.query(
                    "dnase==True").query('media=="cells"').query('dilution == 1e2'), ax=ax[1], errwidth=1)

ax[1].set_ylim((5,8.2))
ax[1].get_yaxis().set_ticks([5,6,7,8])
ax[1].set_ylabel('log10(VG/uL)')
ax[1].set_xlabel('')
ax[1].set_xticklabels(['WT',u'MAAP\n27∆Start\n', u'MAAP\n59∆Stop', u'AAP\n188∆Stop'])
ax[1].legend(frameon=False, loc='upper right', bbox_to_anchor=(1.04, 1.1),handletextpad=.1, labelspacing=.3)
ax[1].yaxis.set_label_coords(-.13, 0.5)

ax2=ax[0]
sns.barplot(x='Name', y='log2_diff', hue='complement_plot',palette=sns.color_palette("Set2"), 
                errwidth=1,
                data=competition_mut_validation_df, ax=ax[2])

ax[2].set_ylim([-10.5,7.5])
ax[2].get_yaxis().set_ticks([-10,-5,0,5])
# ax[2].set_yticks([0], minor=True)
ax[2].axhline(0, c='slategray',alpha=.3, lw=.5)
ax[2].get_yaxis().grid(which='minor')

ax[2].set_ylabel('log2(s\')')
ax[2].set_xlabel('')
ax[2].set_xticklabels([u'MAAP\n27∆Start\n', u'MAAP\n59∆Stop',  u'MAAP\n83∆Stop', u'AAP\n188∆Stop'])
ax[2].legend(frameon=False, loc='upper right', 
             bbox_to_anchor=(1.04, 1.1),handletextpad=.1, labelspacing=.3)

ax[2].yaxis.set_label_coords(-.13, 0.5)
ax[0].set_ylim((-1.1,.75))

# save_fig(fig,"a04_reviison_library_and_validation_complement.pdf" )


# In[27]:


competition_mut_validation_df.groupby(['Name', 'complement'])['log2_diff'].describe()

for mutant in competition_mut_validation_df['Name'].unique():
        log2_diff_pRep = competition_mut_validation_df.            query("Name == @mutant").query("complement == 'pRep'")['log2_diff']
        log2_diff_pRep_maap = competition_mut_validation_df.            query("Name == @mutant").query("complement == 'pRep-MAAP'")['log2_diff']
        print (mutant)
        ttest = stats.ttest_ind(log2_diff_pRep, log2_diff_pRep_maap)
        print (ttest)


# In[28]:


maap_pos_only_diff_drop_log2 = maap_pos_only_diff_drop.stack().apply(np.log2).reset_index()
maap_pos_only_diff_drop_log2.head()


# In[29]:


maap_pos_only_diff_drop_log2.query("abs_pos==83")


# In[30]:


# maap_aa_has_stop_subset_copy_tidy = maap_aa_has_stop_subset_copy.stack().to_frame().reset_index()
# maap_aa_has_stop_subset_copy_tidy['virus_stop'] = maap_aa_has_stop_subset_copy_tidy['stop_maap'] + '-' +maap_aa_has_stop_subset_copy_tidy['level_4']

# sns.barplot(x='level_4', y=0, hue='stop_gee')


# In[31]:


maap_aa_has_stop_subset_copy.query('abs_pos > 34').head()


# In[32]:


maap_aa_has_stop_subset_copy.query("abs_pos==59").query('stop_maap==1').apply(np.log2)


# In[33]:


maap_aa_has_stop_subset_copy['diff'] =  (maap_aa_has_stop_subset_copy['pEK_254']-maap_aa_has_stop_subset_copy['pRep'] ).apply(np.log2)

maap_aa_has_stop_subset_copy.sort_values('diff', ascending=False).query('stop_maap == 1').query('abs_pos>25').query('abs_pos < 130').head()


# In[34]:


maap_aa_has_stop_subset_pos_avg= maap_aa_has_stop_subset_copy.loc[np.arange(0,735)].groupby(level=['abs_pos', 'stop_maap']).mean()
maap_aa_has_stop_subset_pos_avg.head()


# ### shuffle codons within a position and amino acid 
# This gives us the null distribution for non-stop vs stop-codon effect, which we can compare to for p-value computing

# In[35]:


import time
st = time.time()
def shuffle_codons_in_aa_pos(df,num_iter,seed=100,load=False, 
                                 save_path='../data/dataframes/shuffled_codons_df.csv.gz'):
    if load:
        df = pd.read_csv(save_path, index_col=[0,1,2,3], header=[0,1,2,3,4])
        return df.sort_index(axis=1)
    np.random.seed(100)
    for iter_n in range(0, num_iter):
        maap_aa_has_stop_col_subset = df
        maap_aa_has_stop_col_subset_shuffled = maap_aa_has_stop_col_subset.groupby(level= 
            ['abs_pos', 'aa']).transform(np.random.permutation)
        new_cols =  [x + (str(iter_n),) for x in maap_aa_has_stop_col_subset_shuffled.columns ]
        maap_aa_has_stop_col_subset_shuffled.columns = pd.MultiIndex.from_tuples(new_cols)
        
        if iter_n ==0:
            merged_df = maap_aa_has_stop_col_subset_shuffled
        else:    
            merged_df = pd.concat([merged_df,maap_aa_has_stop_col_subset_shuffled ], axis=1)
        
    merged_df.columns.rename(
        ['name', 'comp', 'mut', 'rep', 'random_rep'],inplace=True)
    if save_path:
        merged_df.to_csv(save_path,compression='gzip')
    return merged_df.sort_index(axis=1)

num_iter = 100 # note: p-value from paper was calculated with num_iter = 5000
boostrap_df = shuffle_codons_in_aa_pos(maap_aa_has_stop_subset,num_iter = num_iter,load=True)
print ('time to run %.03f' % ((time.time()-st)/60.0))
boostrap_df.query("abs_pos == 32").head(6)


# In[36]:


boostrap_df_aa_mean = boostrap_df.groupby(level=['abs_pos','aa','stop_maap']).mean().apply(np.log10)

boostrap_df_aa_mean_diff = boostrap_df_aa_mean.unstack().groupby(
    level=['name', 'random_rep'], axis=1).diff(axis=1).xs(1, level='stop_maap', axis=1, drop_level=False)
boostrap_df_aa_mean_diff.query("abs_pos == 27")


# In[37]:


maap_stop_aa_mean = maap_aa_has_stop_subset.groupby(['abs_pos','aa', 'stop_maap']).mean().apply(
    np.log10)
maap_stop_aa_mean_diff = maap_stop_aa_mean.unstack().groupby(
        level=[0],axis=1).diff(axis=1).xs(1, level='stop_maap', axis=1, drop_level=False)
maap_stop_aa_mean_diff.query("abs_pos == 27").head()


# In[38]:


boostrap_df_aa_mean_diff.dropna(how='all').shape


# In[39]:


boostrap_df_aa_mean_diff.loc[maap_stop_aa_mean_diff.index].shape


# In[40]:


maap_stop_aa_mean_diff.dropna(how='all').shape


# In[41]:


maap_stop_aa_mean_diff.shape


# ### using the distribuiton from shuffled codons, calculate z-scores and p-values

# In[42]:


def compute_boostrap_pvalues(bootsrap_diff_df, gene_diff_df):
    bootsrap_diff_df = bootsrap_diff_df.loc[gene_diff_df.index]
    bootsrap_diff_df = bootsrap_diff_df.dropna(how='all')
    gene_diff_df = gene_diff_df.dropna(how='all')
    assert bootsrap_diff_df.shape[0] == gene_diff_df.shape[0]
    no_comp_df = bootsrap_diff_df.xs('pRep',level=0,axis=1,drop_level=False)
    no_comp = bootsrap_diff_df.xs('pRep',level=0,axis=1,drop_level=False).apply(
        lambda x: np.where(gene_diff_df.iloc[:,1].values.flatten()>x, 1,0)).sum(axis=1)
    df=pd.DataFrame()
    df_all_zscores_no_complement = pd.concat(
        [no_comp_df,gene_diff_df.iloc[:,1] ], axis=1).apply(stats.zscore, axis=1)
    df['zscore_nocomp'] = pd.DataFrame(df_all_zscores_no_complement.values.tolist(), index=no_comp_df.index ).iloc[:,-1]
    df['zscore_nocomp_pval'] = df['zscore_nocomp'].apply(special.ndtr)
#     df['zscore_nocomp'] = pd.concat(
#         [no_comp_df,gene_diff_df.iloc[:,1] ], axis=1).apply(stats.zscore, axis=1).iloc[:,-1].values
#     df['zscore_nocomp_pval'] = df['zscore_nocomp'].apply(special.ndtr)
    comp_df = bootsrap_diff_df.xs('pEK_254',level=0,axis=1,drop_level=False)
    comp = bootsrap_diff_df.xs('pEK_254',level=0,axis=1,drop_level=False).apply(
        lambda x: np.where(gene_diff_df.iloc[:,0].values.flatten()>x, 1,0)).sum(axis=1)
    df_all_zscores_complement = pd.concat(
        [comp_df,gene_diff_df.iloc[:,0] ], axis=1).apply(stats.zscore, axis=1)
    df['zscore_comp'] = pd.DataFrame(df_all_zscores_complement.values.tolist(), index=comp_df.index ).iloc[:,-1]
    df['zscore_comp_pval'] = df['zscore_comp'].apply(special.ndtr)
    maap_stat_df = pd.concat([no_comp,comp],axis=1)
#     return maap_stat_df
    maap_pvals = (((maap_stat_df +1) / (no_comp.max()+1)))
#     return df
#     return maap_pvals
    maap_pvals = pd.concat([maap_pvals,df],axis=1)
    return maap_pvals
maap_pvals = compute_boostrap_pvalues(boostrap_df_aa_mean_diff, maap_stop_aa_mean_diff)
maap_pvals.head()


# In[43]:


bootstrap_diff_pos_mean = boostrap_df_aa_mean_diff.groupby(level='abs_pos').mean()
maap_diff_pos_mean = maap_stop_aa_mean_diff.groupby(level='abs_pos').mean()

bootstrap_diff_pos_mean_r10 = bootstrap_diff_pos_mean.rolling(100, min_periods=0).mean()
maap_diff_pos_mean_r10 = maap_diff_pos_mean.rolling(100, min_periods=0).mean()


# In[44]:


maap_pos_pvalues = compute_boostrap_pvalues(bootstrap_diff_pos_mean, maap_diff_pos_mean)
# maap_pos_pvalues.query('abs_pos==32')


# In[45]:


bootstrap_diff_pos_mean_r10 = bootstrap_diff_pos_mean.rolling(30, min_periods=0).mean()
maap_diff_pos_mean_r10 = maap_diff_pos_mean.rolling(30, min_periods=0).mean()
maap_pos_pvalues_rolling = compute_boostrap_pvalues(
    bootstrap_diff_pos_mean_r10,maap_diff_pos_mean_r10)


# ### plot fitness as well as p-value at each position on the capsid

# In[46]:


def plot_fitness_diff( maap_aa_has_stop_subset_pos_avg, column,ax=None, ):
    if not ax:
        fig,ax = plt.subplots(figsize=[8,.5])
    ax.scatter(list(maap_aa_has_stop_subset_pos_avg.query('stop_maap ==  1').reset_index()['abs_pos']),
             maap_aa_has_stop_subset_pos_avg.query('stop_maap ==  1')[column].apply(np.log2),s=.1, c='lightcoral' )
    ax.scatter(list(maap_aa_has_stop_subset_pos_avg.query('stop_maap ==  1').reset_index()['abs_pos']),
             maap_aa_has_stop_subset_pos_avg.query('stop_maap ==  0')[column].apply(np.log2),s=.1,c='slategrey' )

    ax.plot(list(maap_aa_has_stop_subset_pos_avg.query('stop_maap ==  1').reset_index()['abs_pos']),
             maap_aa_has_stop_subset_pos_avg.query(
                 'stop_maap ==  1')[column].rolling(10,min_periods=1).mean().apply(np.log2),c='lightcoral', lw=.5 )

    ax.plot(list(maap_aa_has_stop_subset_pos_avg.query('stop_maap ==  0').reset_index()['abs_pos']),
             maap_aa_has_stop_subset_pos_avg.query(
                 'stop_maap ==  0')[column].rolling(10,min_periods=1).mean().apply(np.log2),c='slategrey',lw=.5  )
    ax.set_xlim(0,735)
    ax.set_ylim(-3.5,.5)
    ax.set_xticks([])
    if not ax:
        return fig


# In[47]:


def plot_pvalues(pval_df,smooth=None,
                 figname=None,
                 zscore=False,
                 figsize=[8,1.2], 
                 b_t_a = [.5,.9], complement=True, all_cond=False):
    fig=plt.figure(figsize=[20,5])
    if all_cond:
        fig, (a0,a1,a2,a3,a4,a5) = plt.subplots(6,1,figsize=figsize,sharex=True ,
            gridspec_kw = {'height_ratios':[.1, 1,1,1,1,1],"hspace":0.005})
        
    elif complement:
        fig, (a0,a1,a2,a3) = plt.subplots(4,1,figsize=figsize,sharex=True ,
            gridspec_kw = {'height_ratios':[.1, 1,1,1],"hspace":0.005})
    else: 
        fig, (a0,a1,a2) = plt.subplots(3,1,figsize=figsize,sharex=True ,
            gridspec_kw = {'height_ratios':[.1, 1,1],"hspace":0.005})
   
    a0.set_xlim(1,735)
    a0.set_ylim(-1,1)
    a0.plot([27,146],[0,0],linewidth=3,color='black', alpha=.65)
    a0.plot([176,380],[0,0],linewidth=3,color='grey', alpha=.65)
    a0.plot([576,732],[0,0],linewidth=3,color='grey', alpha=.65)
    plot_fitness_diff(maap_aa_has_stop_subset_pos_avg,'pRep',a1 )
    if complement:
        plot_fitness_diff(maap_aa_has_stop_subset_pos_avg,'pEK_254',a2 )
#     a1.set_ylabel()
    if all_cond:
        plot_fitness_diff(maap_aa_has_stop_subset_pos_avg,'pEK_256',a3 )
        plot_fitness_diff(maap_aa_has_stop_subset_pos_avg,'pEK_258',a4 )

    if zscore:
        comp_plot_var = 'zscore_comp_pval'
        no_comp_plot_var = 'zscore_nocomp_pval'
    else:
        comp_plot_var = 1
        no_comp_plot_var = 0
    if smooth:
        print ("smoothing")
        if complement:
            plt.plot(pval_df['abs_pos'], pval_df[no_comp_plot_var].apply(
                np.log10).rolling(smooth,min_periods=0).mean(),
                 label='Rep Only',linewidth=.5,c='black')
            plt.plot(pval_df['abs_pos'], pval_df[comp_plot_var].apply(
                np.log10).rolling(smooth,min_periods=0).mean(),
                 label='Rep + MAAP',linewidth=.5,c='blue', alpha=.5,linestyle='dashed')
        else:
            a2.plot(pval_df['abs_pos'], pval_df[no_comp_plot_var].apply(
                np.log10).rolling(smooth,min_periods=0).mean(),
                 label='Rep Only',linewidth=.5,c='black')
            
    else:
        print ("not smoothing")
        if all_cond:
            print( "all conditions")
            plt.plot(pval_df['abs_pos'], pval_df[no_comp_plot_var].apply(np.log10),
                 label='Rep Only',linewidth=.5,c='black')
            plt.plot(pval_df['abs_pos'], pval_df[comp_plot_var].apply(np.log10),
                 label='Rep + MAAP',linewidth=.5,c='blue', alpha=.5,linestyle='dashed')
            plt.plot(pval_df['abs_pos'], pval_df[comp_plot_var].apply(np.log10),
                 label='Rep + MAAP',linewidth=.5,c='blue', alpha=.5,linestyle='dashed')
            plt.ylim((-40,0))
        if complement:
            plt.plot(pval_df['abs_pos'], pval_df[no_comp_plot_var].apply(np.log10),
                     label='Rep Only',linewidth=.5,c='black')
            plt.plot(pval_df['abs_pos'], pval_df[comp_plot_var].apply(np.log10),
                     label='Rep + MAAP',linewidth=.5,c='blue', alpha=.5,linestyle='dashed')
            plt.ylim((-5,0))
        else:
            print ("no comp")
            a2.plot(pval_df['abs_pos'], pval_df[no_comp_plot_var].apply(np.log10),
                     label='Rep Only',linewidth=.5,c='black')
            a2.set_ylim((-40,0))

    plt.legend(bbox_to_anchor=b_t_a,frameon=False)
    a0.set_axis_off()
    plt.xlim(0,735)
#     plt.ylim(-4,0)
#     plt.yticks([0,-1,-2,-3])
    plt.xlabel('VP Position')
#     plt.ylabel('log10(p-val)')
    if figname:
        save_fig(fig, figname)


# In[48]:


## note: to exactly reproduce figure in paper, you will need to run shuffle_codons_in_aa_pos() with num_iter=5000
## this will take >10 hours on standard machine and create 10gb file
plot_pvalues(maap_pos_pvalues_rolling.reset_index(),zscore=True,
             b_t_a=[1.02,.93],figsize=[4,(1.2*.66)],complement=False)

