
# coding: utf-8

# # Thermostability and antibody neutralization analysis

# In[1]:

import os 
import sys 

from Bio import AlignIO
import pandas as pd 
pd_idx =pd.IndexSlice
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr


from common import mm_to_inch,load_axis_contacts
get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')
sys.path.append('../x01_process_data/')
import x02_load_dataframes
import x03_compute_selections

sys.path.append('../settings/')
# from paths_new import FIGURES_DIR
from common import DESIRED_AA_ORD
from common import codon_to_aa_selector
from paper_settings import PAPER_PRESET
from paper_settings import PAPER_FONTSIZE
from paper_settings import save_fig


sns.set(**PAPER_PRESET)


# In[2]:

package_data = x02_load_dataframes.load_packaging_df()
tm_data = x02_load_dataframes.load_thermostability_df()


# In[3]:

tm_selection_dict = x03_compute_selections.compute_tm_selection(package_data, tm_data,wt_norm=True,sum_all=True)


# In[4]:

tma_aa_selection = tm_selection_dict['aa_selection']
tma_barcode_selection = tm_selection_dict['barcode_selection']
tma_aa_mean = tma_aa_selection.loc[:,pd_idx[:,['67']]].groupby(level='virus', axis=1).mean()
tma_aa_mean.head()


# In[5]:

package_aa_sel = x03_compute_selections.compute_packaging_selection(package_data, level='aa')
package_barcode_sel = x03_compute_selections.compute_packaging_selection(package_data, level='barcode')
tm_aa_filtered = codon_to_aa_selector(
    package_aa_sel[package_aa_sel[('CMV','0')]>.5], tma_aa_selection)

package_aa_sel_filtered = codon_to_aa_selector(
    package_aa_sel[package_aa_sel[('CMV','0')]>.5], package_aa_sel)

tm_barcode_filtered = codon_to_aa_selector(
    package_aa_sel[package_aa_sel[('CMV','0')]>.5], tma_barcode_selection)



# In[6]:

package_aa_sel_filtered_all = package_aa_sel_filtered.iloc[:,:2]
package_aa_sel_filtered_all.columns = package_aa_sel_filtered_all.columns.droplevel(1)
package_aa_sel_filtered_all.index = package_aa_sel_filtered_all.index.droplevel(2)
package_aa_sel_filtered_all.head()


# In[7]:

tma_aa_mean.index = tma_aa_mean.index.droplevel(2)
tma_aa_mean.head()


# In[8]:

tm_package_compare = package_aa_sel_filtered_all.join(tma_aa_mean,how='inner')
tm_package_compare.dropna(inplace=True)
tm_package_compare.head()


# ### histogram of selection values across temperatures

# In[9]:

tms_good = ['55','60','65','66','67','68']
fig, axes = plt.subplots(ncols=1, nrows=6, sharex=True, sharey=True, figsize=(2,7))
for tm, ax in zip(tms_good, axes.flatten()):
    tm_in = tm_aa_filtered.groupby(
    level=['abs_pos', 'aa', 'wt_bc']).mean().xs(
        ('CMV2', tm), level=['virus','tm'], axis=1).dropna().apply(np.log2)
    count = tm_in[tm_in<-2.5].count()
    percent_in_tail = (float(count) / tm_in.count() ).values[0]
    sns.distplot(tm_aa_filtered.groupby(
    level=['abs_pos', 'aa', 'wt_bc']).mean().xs(
        ('CMV2', tm), level=['virus','tm'], axis=1).dropna().apply(np.log2),ax=ax,axlabel="Temp: %sC" % (tm),color='slategray')
    ax.plot([-2.5,-2.5],[0,1],c='grey')
    ax.text(-7,0.1,"%.01f%%" % (percent_in_tail*100),size=7)
    ax.set_ylim(0,1)
    ax.set_yticks([0,1])
plt.tight_layout()


# ### positional frequency of mutations in the left tail of selection distribution - less than log2(-2.5) 

# In[10]:

tm_aa_left_tail = tm_aa_filtered[tm_aa_filtered < (2**-2.5)]
fig, axes = plt.subplots( nrows=6, sharex=True, sharey=True, figsize=(5,7))
for tm, ax in zip(tms_good, axes.flatten()):
    ax.plot(tm_aa_left_tail.groupby(level='abs_pos').apply(
        lambda x: x.count()/len(x)).xs(('CMV2', tm), level=['virus', 'tm'], axis=1),c='slategray',linewidth=.5)
    ax.set_xlabel("Temp: %sC" % (tm))
plt.tight_layout()


# ### focus on temperature 65C, the measured Tm for AAV2

# In[11]:

tm_65_percents = tm_aa_left_tail.groupby(level='abs_pos').apply(
        lambda x: x.count()/len(x)).xs(('CMV2', tm), level=['virus', 'tm'], axis=1).reset_index()
tm_65_percents.columns = tm_65_percents.columns.droplevel(1)
tm_65_percents_subs = tm_65_percents[tm_65_percents['abs_pos'].isin(np.arange(0,735))]
tm_65_percents_subs['abs_pos']  = tm_65_percents_subs['abs_pos'].apply(int)


# ### load known AAV2 contatcs from VIPERDB 
# add selection values for each contact  
# website: http://viperdb.scripps.edu/

# In[12]:

contacts_df = load_axis_contacts()
contacts_df_with_tm_freq =  contacts_df.merge(
    tm_65_percents_subs, left_on='res1_vp', right_on='abs_pos').merge(
    tm_65_percents_subs, left_on='res2_vp', right_on='abs_pos')
contacts_df_with_tm_freq.sort_values('CMV2_x',ascending = False).head()


# subset contacts at two-fold axis

# In[13]:

two_fold = contacts_df_with_tm_freq[['A1-A6 (I-2)', 'CMV2_x', 'CMV2_y', 'abs_pos_x', 'abs_pos_y']].dropna()
two_fold_deduped = pd.concat([pd.concat([two_fold['CMV2_x'],two_fold['CMV2_y']] ), 
pd.concat([two_fold['abs_pos_x'],two_fold['abs_pos_y']] )],axis=1).drop_duplicates()
two_fold_deduped.head()


# subset three-fold axis

# In[14]:

three_fold = contacts_df_with_tm_freq[['A1-A7 (I-3)', 'CMV2_x', 'CMV2_y', 'abs_pos_x', 'abs_pos_y']].dropna()


# In[15]:

three_fold_deduped = pd.concat([pd.concat([three_fold['CMV2_x'],three_fold['CMV2_y']] ), 
pd.concat([three_fold['abs_pos_x'],three_fold['abs_pos_y']] )],axis=1).drop_duplicates()
three_fold_deduped.head()


# subset five-fold axis

# In[16]:

five_fold_1 = contacts_df_with_tm_freq[['A1-A2 (I-5)', 'CMV2_x', 'CMV2_y', 'abs_pos_x', 'abs_pos_y']].dropna()
five_fold_1.head()


# In[17]:

five_fold_deduped = pd.concat([pd.concat([five_fold_1['CMV2_x'],five_fold_1['CMV2_y']] ), 
pd.concat([five_fold_1['abs_pos_x'],five_fold_1['abs_pos_y']] )],axis=1).drop_duplicates()
five_fold_deduped.head()


# ### P-Value 3-fold contacts vs all other positions

# first subset three-fold vs everything else

# In[18]:

not_three_fold = tm_65_percents_subs[~tm_65_percents_subs['abs_pos'].isin(three_fold_deduped[1])]
not_three_fold.head()


# In[19]:

not_three_fold['CMV2'].mean() - three_fold_deduped[0].mean()


# In[20]:

not_three_fold_proportion = (len(not_three_fold.loc[not_three_fold['CMV2'] > 0]) / len(not_three_fold))
three_fold_proportion = len(three_fold_deduped.loc[three_fold_deduped[0] > 0]) / len(three_fold_deduped)
print (three_fold_proportion)
print (not_three_fold_proportion)
three_fold_proportion/not_three_fold_proportion


# In[21]:

import scipy.stats as stats
def two_proprotions_confint(success_a, size_a, success_b, size_b, significance = 0.05):
    """
    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group

    size_a, size_b : int
        Size, or number of observations in each group

    significance : float, default 0.05

    Returns
    -------
    prop_diff : float
        Difference between the two proportion

    confint : 1d ndarray
        Confidence interval of the two proportion test
    """
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    var = prop_a * (1 - prop_a) / size_a + prop_b * (1 - prop_b) / size_b
    se = np.sqrt(var)

    # z critical value
    confidence = 1 - significance
    z = stats.norm(loc = 0, scale = 1).ppf(confidence + significance / 2)

    # standard formula for the confidence interval
    # point-estimtate +- z * standard-error
    prop_diff = prop_b - prop_a
    confint = prop_diff + np.array([-1, 1]) * z * se
    return prop_diff, confint


# In[22]:

null_obs = len(not_three_fold)
null_successes = len(not_three_fold.loc[not_three_fold['CMV2'] > 0]) 
threefold_obs = len(three_fold_deduped)
threefold_sucesses = len(three_fold_deduped.loc[three_fold_deduped[0] > 0])

two_proprotions_confint(success_a=threefold_sucesses, 
                        size_a=threefold_obs, 
                        success_b=null_successes, 
                        size_b=null_obs, 
                        significance = 0.00000000000001)


# ### Antibody Analysis

# In[23]:

antibody_counts = x02_load_dataframes.load_antibody_df()
antibody_counts.head()


# In[24]:

antibody_selection_df= x03_compute_selections.compute_antibody_selection(
            ab_counts=antibody_counts, package_counts=package_data, wt_norm=True)
antibody_selection_df.head()


# ### histogram of a20 interacting residues vs not 

# In[25]:

a20_postions_subs = np.array([261,262,263,264,384,385,708,717,258,253,254,658,659,660,548,556])

def plot_a20_dist(ab_aa_df,
#                   hek_or_cre='HEK',
                  plot_wt_random = True,
                  plot_zoom=False,
                  legend_on=True,
                  figname=None, 
                  return_values=False):

    fig = plt.figure(figsize=[1.5,1.5])
    if plot_wt_random:
        wt = ab_aa_df.mean(axis=1).apply(np.log2).dropna()
        sns.kdeplot(ab_aa_df.mean(axis=1).apply(np.log2).dropna(), label='Other',
                    alpha=.9,color='gray', **{'linestyle':'dashed'})
    a20 = ab_aa_df.loc[a20_postions_subs].mean(axis=1).apply(np.log2).dropna()
    sns.kdeplot(ab_aa_df.loc[a20_postions_subs].mean(axis=1).apply(np.log2).dropna(), label='A20')
    print ('mannwhitney test pval = %s' % stats.mannwhitneyu(wt,a20)[1])
    print ('effect size: %s' % (wt.mean()- a20.mean()) )
    plt.xlabel("")
    plt.ylabel('')
    plt.plot([2.5,2.5],[0,2],c='black',lw=.5)
    if plot_zoom:
        plt.ylim([0,.5])
        plt.yticks([])
    if legend_on:
        plt.legend().set_visible(legend_on)
        plt.legend(bbox_to_anchor=(.4825,1),frameon=False)
    else:
        plt.legend().set_visible(legend_on)
    plt.tight_layout()
    if figname:
        figpath =  os.path.join(FIGURES_DIR, figname)
        save_fig(fig, figpath)
    if return_values:
        return wt, a20

plot_a20_dist(antibody_selection_df,figname = None)


# In[26]:

wt, a20 = plot_a20_dist(antibody_selection_df,plot_zoom=True,legend_on=False, 
              figname = None, return_values=True)


# ### effect size and p-value for known a20 interacting positions vs not

# In[27]:

wt_successes = len(wt[wt > 2.5])
wt_obs = len(wt)
a20_successes = len(a20[a20>2.5])
a20_obs = len(a20)
proportion_a20 =   a20_successes / a20_obs
proportion_wt =   wt_successes / wt_obs
print ('proportion wt: %s' %proportion_wt )
print ('proportion a20: %s' %proportion_a20 )
print ('effect size: %s' % (proportion_a20 / proportion_wt))


# In[28]:

two_proprotions_confint(success_a=wt_successes,
                        size_a=wt_obs, 
                        success_b=a20_successes, 
                        size_b=a20_obs, 
                        significance = 1e-16)


# ### heatmap of selection values for positions known to interact with a20

# In[29]:

antibody_selection_mean = antibody_selection_df.mean(axis=1)
antibody_selection_mean.head()


# In[30]:

antibody_selection_mean_for_heatmap = antibody_selection_mean.unstack(0).query("wt_bc==0").query("lib_type.isin(['sub','del'])")
antibody_selection_mean_for_heatmap = antibody_selection_mean_for_heatmap.loc[:,a20_postions_subs]
antibody_selection_mean_for_heatmap.index = antibody_selection_mean_for_heatmap.index.droplevel([1,2])
antibody_selection_mean_for_heatmap = antibody_selection_mean_for_heatmap.reindex(DESIRED_AA_ORD)
antibody_selection_mean_for_heatmap.head()


# In[31]:

sns.set(**PAPER_PRESET)
fig,ax =plt.subplots(figsize=[3.174*1.5, 1.625*1.5])
sns.heatmap(antibody_selection_mean_for_heatmap.apply(np.log2), cmap='RdBu_r', vmin=-10, vmax=10, ax=ax, yticklabels=True)
ax.set_ylabel("Amino Acid")
ax.set_xlabel("VP Position")


# In[32]:

all_residues = antibody_selection_mean.apply(np.log2).dropna()
a20_residues = antibody_selection_mean.loc[a20_postions_subs].apply(np.log2).dropna()

a20_fraction = a20[a20 > 2.5].count() / float(a20.count())
all_residues_fraction = all_residues[all_residues > 2.5].count() / float(all_residues.count())
fig = plt.figure(figsize=(1.2,1.1))
plt.bar(['A20\nInteracting','Other\nResidues'],
        [a20_fraction,
         all_residues_fraction],color='slategrey')
plt.ylabel('fraction > cutoff')

