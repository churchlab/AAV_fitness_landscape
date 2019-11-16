
# coding: utf-8

# In[1]:

from __future__ import print_function
import pandas as pd 
pd_idx =pd.IndexSlice
import numpy as np

import x02_load_dataframes


# In[2]:

package_counts = x02_load_dataframes.load_packaging_df()
mouse_counts = x02_load_dataframes.load_mouse_df()
# ab_counts =x02_load_dataframes.load_antibody_df()
# tm_counts = x02_load_dataframes.load_thermostability_df()
# maap_counts = x02_load_dataframes.load_MAAP_df()
# package_counts.head()


# In[3]:

def drop_bad_reps(package_df):
    return package_df.drop(labels=[('CMV', 'virus', '1', 'a','count', 'GEN00105095'),
                                   ('CMV', 'virus', '2', 'a','count',  'GEN00105096'),
                                   ('Rep', 'virus', '4', 'c', 'count', 'GEN00105094'),
                                   ('Rep', 'virus', '3', 'c', 'count', 'GEN00105093')], axis=1)
    


# In[4]:

def compute_packaging_selection(df_in=None,
                      level='barcode',
                      wt_norm=True,
                      sum_measurments=True,
                      sum_techincal_replicates= True,
                      for_plotting=True,
                     drop=True):
    '''
    Takes packaging df and returns same size df but with selection values
    
    Args:
    df_in - this is packaging df 
    level - either 'barcode', 'codon', or 'aa': chooses where to sum the measurments 
    sum_measurments: chooses whether to sum measurments from different tiles (bbsI vs BsmbI)
    for_plotting: if true, sums to the simplest form for plotting thus that all measurments which makup up a codon
    or aa are given as one data point
    
    return df - a df with selection values, based on the above inputs 
    '''
    ##drop replicates with sparse data, these were prepped different (using qiagen PCR purification kit)
    if drop:
        df_in = drop_bad_reps(df_in)
    if sum_techincal_replicates:
        df_in = df_in.groupby(level=[0,1,2], axis=1).sum()
    else: 
        df_in = df_in.groupby(level=[0,1,2,3,4], axis=1).sum()
    if for_plotting:

        if level =='aa':
            wt = 'is_wt_aa'
            index = ['abs_pos', 'aa','is_wt_aa','lib_type']
        if level =='codon':
            wt= 'is_wt_codon'
            index = ['abs_pos', 'aa', 'codon','aa-codon','lib_type','is_wt_codon']
        if level == 'barcode':
            wt ='wt_bc'
            index= ['abs_pos', 'aa', 'codon','lib_type','barcode', 'wt_bc']
            
        df_in_g = df_in.groupby(level=index).sum()
    
    else:
        if level =='aa':
            wt='is_wt_aa'
            index = 8
        if level =='codon':
            wt='is_wt_codon'
            index = 10
        if level == 'barcode':
            wt='wt_bc'
            index=11
        df_in_g = df_in.groupby(level=list(range(index))).sum()
        
    df = pd.DataFrame(index=df_in_g.index)

    for col in df_in_g.loc[:,pd_idx[:,'plasmid',:]]:
        promoter =  col[0]
        lib = col[2]

        if lib == '0':
            freq_p =  df_in_g.loc[
                :,pd_idx[promoter,'plasmid',:]].apply(lambda x: x/np.nansum(x))
            if sum_measurments:
                freq_v = pd.DataFrame(df_in_g.loc[
                    :,pd_idx[promoter,'virus',['1','2','3','4'],:]].sum(axis=1)).apply(lambda x: x/np.nansum(x))
            else:
                
                if sum_measurments:
                    freq_v = pd.DataFrame(df_in_g.loc[
                        :,pd_idx[promoter,'virus',['1','2','3','4'],:]].sum(axis=1)).apply(lambda x: x/np.nansum(x))
                else:
                    freq_v = df_in_g.loc[
                        :,pd_idx[promoter,'virus',['1','2','3','4'],:]].apply(lambda x: x/np.nansum(x))
            selection = freq_v.div(freq_p.values[:,0],axis=0)
            if sum_measurments:
                df[(promoter, lib)] = freq_v.div(freq_p.values[:,0],axis=0)
            else:
                df = pd.concat([df, selection], axis=1)

        else:
            freq_p =  df_in_g.loc[:,pd_idx[promoter,'plasmid',lib,:]].apply(lambda x: x/np.nansum(x))
            freq_v = df_in_g.loc[:,pd_idx[promoter,'virus',lib,:]].apply(lambda x: x/np.nansum(x))
            selection = freq_v.div(freq_p.values[:,0],axis=0)
            if sum_measurments:
                df[(promoter, lib)] = freq_v.div(freq_p.values[:,0],axis=0)
            else:
                df = pd.concat([df, selection], axis=1)
    
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.replace([np.inf,-np.inf,0], np.nan)
    if wt_norm:
        df = df.div(df.xs(1, level=wt).mean())
    return df

# p_df = compute_packaging_selection(
#     package_counts,level='codon', wt_norm=True, sum_measurments=False, for_plotting=False)
# p_df


# In[5]:

def compute_mouse_selections(package_counts, 
                             mouse_counts,
                             wt_norm=False,
                             drop=True,
                             drop_tile = True,
                             return_freq=False):
    if drop:
        package_counts = drop_bad_reps(package_counts)
    
    new_mouse_selection = mouse_counts
    # one tile is synthesized twice (with different enzymes), 
        #dropping this level allows us to compute aa selecttion
    if drop_tile: 
        mouse_counts.index = mouse_counts.index.droplevel(1)
        package_counts.index = package_counts.index.droplevel(1)
    
    #compute freq of packaging data for aa and barcode level
    # we do this sperately for the two virus production reps (CMV1 & CMV2) since we inject both of them in different mice
    CMV12_barcode_summed = package_counts.loc[:, pd_idx['CMV' , :]].sum(axis=1)
    CMV12_barcode_summed_freq = CMV12_barcode_summed / CMV12_barcode_summed.sum()
    CMV_barcode_freq = package_counts.xs(
        ('CMV','0'), level =['promoter', 'virus_rep'],axis=1 ).values.flatten()
    CMV1_barcode_sum = package_counts.loc[
            :,pd_idx['CMV', :,'1']].sum(axis=1)
    CMV1_barcode_freq = CMV1_barcode_sum / np.nansum(CMV1_barcode_sum)
    CMV2_barcode_sum = package_counts.loc[
            :,pd_idx['CMV',:,'2']].sum(axis=1)
    CMV2_barcode_freq = CMV2_barcode_sum / np.nansum(CMV2_barcode_sum)
    CMV_aa_freq = package_counts.xs(('CMV','0'), level =['promoter', 'virus_rep'],axis=1 ).groupby(
        level=['abs_pos', 'aa','wt_bc', 'lib_type' ]).sum().sum(axis=1)
    CMV_aa_freq = CMV_aa_freq / np.nansum(CMV_aa_freq)
    CMV1_aa_sum = package_counts.xs(('CMV','1'), level =['promoter', 'virus_rep'],axis=1,drop_level=False ).groupby(
    level=['abs_pos', 'aa','wt_bc', 'lib_type' ]).sum().sum(axis=1)
    CMV1_aa_freq= CMV1_aa_sum / np.nansum(CMV1_aa_sum)
    CMV2_aa_sum = package_counts.xs(('CMV','2'), level =['promoter', 'virus_rep'],axis=1,drop_level=False ).groupby(
        level=['abs_pos', 'aa','wt_bc', 'lib_type' ]).sum().sum(axis=1)
    CMV2_aa_freq= CMV2_aa_sum / np.nansum(CMV2_aa_sum)
    CMV12_aa_freq =package_counts.xs(('CMV','virus'), level =['promoter', 'source'],axis=1,drop_level=False ).groupby(
        level=['abs_pos', 'aa','wt_bc', 'lib_type' ]).sum().sum(axis=1)
    CMV12_aa_freq =  CMV12_aa_freq / np.nansum(CMV12_aa_freq)
    
    #compute mouse freq for aa and barcode level
    mouse_counts_aa_summed =  new_mouse_selection.groupby(
        level=['abs_pos',  'aa','wt_bc' ,'lib_type'], axis=0).sum()
    mouse_counts_aa_freq = mouse_counts_aa_summed.apply(lambda x: x/np.nansum(x))
    if return_freq:
        return {'mouse_aa_freq':mouse_counts_aa_freq,
                'CMV1_aa_freq':CMV1_aa_freq, 'CMV2_aa_freq': CMV2_aa_freq,
               'CMV1_barcode_freq':CMV1_barcode_freq, 'CMV2_barcode_freq': CMV2_barcode_freq,
               'CMV1_aa_sum':CMV1_aa_sum, 'CMV2_aa_sum': CMV2_aa_sum,
               'CMV1_barcode_sum':CMV1_barcode_sum, 'CMV2_barcode_sum': CMV2_barcode_sum}
    
    ## compute selection over packaging data
    mouse_counts_barcode_freq = new_mouse_selection.apply(lambda x: x/np.nansum(x))
    mouse_selection_barcode = pd.concat(
        [mouse_counts_barcode_freq.xs('1', level='virus', axis=1, drop_level=False).div(
            CMV1_barcode_freq, axis='index'), 
         mouse_counts_barcode_freq.loc[:, pd_idx[:,'2',: ]].div(
             CMV2_barcode_freq, axis='index')], axis=1)
    mouse_selection_aa = pd.concat(
        [mouse_counts_aa_freq.loc[:, pd_idx[:,'1',: ]].div(
         CMV1_aa_freq, axis='index'), 
         mouse_counts_aa_freq.loc[:, pd_idx[:,'2',: ]].div(
         CMV2_aa_freq, axis='index')], axis=1)
    if wt_norm:
        mouse_selection_barcode = mouse_selection_barcode / mouse_selection_barcode.xs(
            1, level='wt_bc').median()
        mouse_selection_aa = mouse_selection_aa / mouse_selection_aa.xs(1, level='wt_bc').median()
    
    return_dict = {}
    return_dict['barcode_selection'] = mouse_selection_barcode.replace([np.inf, -np.inf,0], np.nan)
    return_dict['aa_selection'] = mouse_selection_aa.replace([np.inf, -np.inf,0], np.nan)
    return return_dict

# mouse_selection_dict = compute_mouse_selections(package_counts, mouse_counts,wt_norm=True)
# mouse_selection_dict['aa_selection'].head()


# In[6]:

def compute_antibody_selection(package_counts, ab_counts,
                               wt_norm=True):
    
    package_counts = drop_bad_reps(package_counts)
    
    #compute freq of packaging data for aa and barcode level
    REP2_aa_freq = package_counts.xs(('Rep','4'), level =['promoter', 'virus_rep'],axis=1,drop_level=False ).groupby(
        level=['abs_pos', 'aa','wt_bc', 'lib_type' ]).sum().sum(axis=1)
    REP2_aa_freq= REP2_aa_freq / np.nansum(REP2_aa_freq)    

    #compute antibody freq for aa 
    ab_counts_aa_summed =  ab_counts.groupby(
        level=['abs_pos',  'aa','wt_bc' ,'lib_type'], axis=0).sum()
    ab_counts_aa_freq = ab_counts_aa_summed.apply(lambda x: x/np.nansum(x))
    ab_selection_aa = ab_counts_aa_freq.div(REP2_aa_freq, axis='index')
    if wt_norm:
        ab_selection_aa = ab_selection_aa / ab_selection_aa.xs(1, level='wt_bc').median()
    return ab_selection_aa.replace([np.inf, -np.inf,0], np.nan)

# ab_dict = compute_antibody_selection(package_counts, ab_counts)
# ab_dict['barcode_selection'].head()


# In[7]:

def compute_tm_selection(package_counts, tm_counts,sum_all=False, wt_norm=False):
    package_counts = drop_bad_reps(package_counts)
    #compute freq of packaging data for aa and barcode level
    REP2_barcode_freq = pd.DataFrame(
        package_counts.loc[
            :,pd_idx['Rep',:,'4']].sum(axis=1)).apply(lambda x: x / np.nansum(x))
    CMV2_barcode_freq = pd.DataFrame(
        package_counts.loc[
            :,pd_idx['CMV',:,'2']].sum(axis=1)).apply(lambda x: x / np.nansum(x))
    REP2_aa_freq = package_counts.xs(('Rep','4'), level =['promoter', 'virus_rep'],axis=1,drop_level=False ).groupby(
        level=['abs_pos', 'aa','wt_bc', 'lib_type' ]).sum().sum(axis=1)
    REP2_aa_freq= REP2_aa_freq / np.nansum(REP2_aa_freq)    
    CMV2_aa_freq = package_counts.xs(('CMV','2'), level =['promoter', 'virus_rep'],axis=1,drop_level=False ).groupby(
        level=['abs_pos', 'aa','wt_bc', 'lib_type' ]).sum().sum(axis=1)
    CMV2_aa_freq= CMV2_aa_freq / np.nansum(CMV2_aa_freq)
    
    if sum_all:
        tm_counts = tm_counts.groupby(level=['virus','tm'], axis=1).sum()
    #compute mouse freq for aa and barcode level
    tm_counts_aa_summed =  tm_counts.groupby(
        level=['abs_pos',  'aa','wt_bc' ,'lib_type'], axis=0).sum()
    tm_counts_aa_freq = tm_counts_aa_summed.apply(lambda x: x/np.nansum(x))
#     return tm_counts_aa_freq, CMV2_aa_freq, REP2_aa_freq
    tm_counts_barcode_freq = tm_counts.apply(lambda x: x/np.nansum(x))
    tm_selection_barcode = pd.concat(
        [tm_counts_barcode_freq.xs('CMV2', level='virus', axis=1, drop_level=False).div(
            CMV2_barcode_freq[0].values, axis='index'), 
         tm_counts_barcode_freq.loc[:, pd_idx['Rep2',: ]].div(
             REP2_barcode_freq[0].values, axis='index')], axis=1)
    tm_selection_aa = pd.concat(
        [tm_counts_aa_freq.loc[:, pd_idx['CMV2',: ]].div(
         CMV2_aa_freq, axis='index'), 
         tm_counts_aa_freq.loc[:, pd_idx['Rep2',: ]].div(
         REP2_aa_freq, axis='index')], axis=1)
    if wt_norm:
        tm_selection_barcode = tm_selection_barcode / tm_selection_barcode.xs(
            1, level='wt_bc').median()
        tm_selection_aa = tm_selection_aa / tm_selection_aa.xs(1, level='wt_bc').median()
    
    return_dict = {}
    return_dict['barcode_selection'] = tm_selection_barcode.replace([np.inf, -np.inf,0], np.nan)
    return_dict['aa_selection'] = tm_selection_aa.replace([np.inf, -np.inf,0], np.nan)
    return return_dict

# tm_selection_dict = compute_tm_selection(package_counts, tm_counts)
# tm_selection_dict['aa_selection'].head()


# In[8]:

def compute_maap_selection( package_counts,maap_counts,wt_norm=True ,level='codon'):
    package_counts = drop_bad_reps(package_counts)
    if level =='codon':
        grouby_level = ['abs_pos', 'aa', 'is_wt_aa', 'is_wt_codon', 'wt_bc', 'lib_type','codon']
    else:
        print ("only codon level selection supported for maap data")
        return
    
    maap_summed = maap_counts.groupby(level=grouby_level).sum()
    plasmid_summed = package_counts.groupby(level=grouby_level).sum().xs(
        ('CMV', 'plasmid',), level=['promoter', 'source'],axis=1).sum(axis=1)
    maap_freq = maap_summed / maap_summed.sum()
    plasmid_freq = plasmid_summed / plasmid_summed.sum()
    maap_selection = maap_freq.div(plasmid_freq,axis='index')
    if wt_norm:
        maap_selection = maap_selection / maap_selection.xs(1,level='wt_bc' ).mean()

    
    return maap_selection.replace([np.inf, -np.inf,0], np.nan)

# maap_selection = compute_maap_selection(package_counts, maap_counts)
# maap_selection.head()


# In[ ]:



