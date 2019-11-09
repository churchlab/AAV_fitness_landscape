#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import json 

sys.path.append('../settings/')
from paths_new import DESIRED_AA_ORD
from paths_new import MOUSE_VAL_DIR
from paths_new import FIGURES_DIR
from paths_new import META_DIR
from paths_new import LOOKUP_DF
WT_PLOTTING_DF = pd.read_csv(os.path.join(META_DIR,'lookup_rc.txt'))
tissue_grabber = ['blood', 'heart', 'kidney', 'lung', 'liver','spleen']
from paper_settings import save_fig
DESIRED_AA_COD_ORD = json.load(open(os.path.join(META_DIR, 'desired_order_aa_codon.json'), 'rb'))
mm_to_inch = 0.0393701


# In[1]:


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

def tidy_mouse_df(df):
    df_tidy = df.apply(np.log2).stack([0,1,2,3,4,5,6]).reset_index()
    df_tidy['aa_gene'] = df_tidy['abs_pos'].apply(str)+'-'+df_tidy['aa']+'-'+df_tidy['lib_type']
    return df_tidy.replace([np.inf, -np.inf], np.nan).dropna()



def swarm_plot_genes(df_tidy,swarm = True, violin = False,boxplot=False,barplot=False,color=None,errwidth=.5):
#     plt.figure(figsize=[4,1])
    if swarm:
        g = sns.swarmplot(x='aa_gene', y=0, hue='organ', data=df_tidy, 
            dodge=True, hue_order=tissue_grabber,palette = sns.color_palette("muted", 10),color=color )
    if violin:
        g = sns.violinplot(x='aa_gene', y=0, hue='organ', data=df_tidy, 
            dodge=True, hue_order=tissue_grabber,palette = sns.color_palette("muted", 10) )
    if boxplot:
        g = sns.boxplot(x='aa_gene', y=0, hue='organ', data=df_tidy, 
            dodge=True, hue_order=tissue_grabber,palette = sns.color_palette("muted", 10),showfliers=False )
    if barplot:
        g = sns.barplot(x='aa_gene', y=0, hue='organ', errwidth=errwidth, data=df_tidy, 
            dodge=True, hue_order=tissue_grabber,palette = sns.color_palette("muted", 10) )
        
    g.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    return g


def plot_best(df_in,mouse_sel_df,  tissue, median,
              codon_or_aa='codon', 
              ratio_or_count ='ratio_nan', 
              ratio_or_count_num = .5,
              selection_number=10,
              swarm=True, 
              boxplot=False,
              violin = False,
              barplot=False, 
              color=None):    
    for tissue in [tissue]:    
        tissue_filtered = df_in[
            (df_in[('%s' % (tissue),'median' )] > median) & 
            (df_in[('%s' % (tissue),ratio_or_count )] >ratio_or_count_num)  ]
        top10 = tissue_filtered.sort_values(
            ('%s' % (tissue),'median'),ascending=False).head(selection_number)
#         print top10['%s_codon' % tissue]
        df = pd.DataFrame()
        mutant_list = []
        for mut in  top10.index.values:
            abs_pos = mut[0] if mut[3] == 'sub' else mut[0] - .5

#             print mut
            df = pd.concat(
                [df, mouse_sel_df.xs(mut, level=['abs_pos', 'aa','wt_bc', 'lib_type'], drop_level=False)])
#             wt_res =  mouse_sel_all_wt_norm.xs(
#                 (abs_pos,1), level=['abs_pos', 'is_wt_codon']).index.get_level_values(1).values[0]
#         print wt_res
            mutant_add = list(mut)
#             mutant_add.insert(0, wt_res)
#         print mutant_add
            mutant_list.append(tuple(mutant_add))
        # df = df.xs('kidney', level='organ', axis=1,drop_level=False)
    #     df.head()
        
#         return df

        df_tidy = tidy_mouse_df(df)
#         return df_tidy
#     return df_tidy  
        g = swarm_plot_genes(df_tidy, swarm=swarm,violin=violin, boxplot=boxplot,barplot=barplot,color=color)
        g.set_title('Best %s Mutants' % tissue, size=20)
        g.legend(bbox_to_anchor=(-.12, 1), loc=2, borderaxespad=0.)
        g.set_xticklabels(['%s-%s-%s' % (x[0],x[1],x[2]) for x in mutant_list], rotation = 45)
        g.set(ylim=(-2,2))
        plt.show()
#         return df 

def pearsonfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("pearson r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

def spearmanfunc(x, y, **kws):
    r, _ = stats.spearmanr(x, y)
    ax = plt.gca()
    ax.annotate("spearman r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

# sample replicates & drop NANs
def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=100, cmap=cmap, **kwargs)

def pairplot_tissue(df, numsamples=None,restriction=None,state=5,alpha=.1,text_size=7,s=1,hex_plt=False, kde=False,height=1.2):
    df_sample=df
    if numsamples:
        df_sample = df.sample(numsamples, axis=1,random_state=state).dropna()
    if restriction:
        df_sample = df_sample[df_sample < restriction].dropna()
    g = sns.PairGrid(df_sample,height=height)
    if hex_plt:
        g.map_upper(plt.hexbin,mincnt=1,cmap='inferno',gridsize=100)
    if kde:
        g.map_upper(sns.kdeplot)
    else:
        g.map_upper(plt.scatter, s=s, marker="o", alpha=alpha)
    g.map_diag(sns.kdeplot)
    g.map_lower(plt.scatter,s=s, marker="o", alpha=alpha)
    g.map_upper(pearsonfunc,size=text_size)
    g.map_lower(spearmanfunc,size=text_size)
    return g 

def tissue_translator(x):
    tissue_dict = {'H':'heart', 'Li':'liver','K':'kidney', 'Lu':'lung', 'Sp':'spleen', 'Br':'brain', 'Bl':'blood'}
    return tissue_dict[x] 

def construct_validation_qPCR_df(meta_df_path_1 = '20180430_mouse_validation_qPCR.csv',
        meta_df_path_2 = '20180501_mouse_validation_qPCR.csv',
        data_df_path_1 = '20180430_PJO_mouse_validation_titer.csv',
        data_df_path_2 = '20180501_Mouse_validation_titer_2.csv',
        mutant_info_path_df = 'mutant_info_df.csv'):
    
    dir_in = os.path.join(MOUSE_VAL_DIR, '%s' )
    meta_df_mouse_val_1 = pd.read_csv(dir_in % meta_df_path_1)
    mutant_info_df = pd.read_csv(dir_in % mutant_info_path_df)
    data_df_mouse_val_1 = pd.read_csv(dir_in%data_df_path_1)
    data_2 = pd.read_csv(dir_in%data_df_path_2)
    meta_2 =  pd.read_csv(dir_in%meta_df_path_2)
    
    data_merge = data_df_mouse_val_1.merge(meta_df_mouse_val_1, on='Well')
    data_2_merge = data_2.merge(meta_2, on='Well')
    
    all_data = pd.concat([data_2_merge, data_merge])
    all_data_cqs = all_data.groupby(
        ['pEK', 'Fluor','tissue', 'biological' ])['Cq'].mean().unstack(1).dropna(axis=1).reset_index()
    all_data_cqs['log2-selection'] = all_data_cqs.FAM - all_data_cqs.HEX
    all_data_cqs_virus = all_data_cqs[all_data_cqs['tissue'] == 'virus ']
    all_data_cqs_tissues = all_data_cqs[all_data_cqs['tissue'] != 'virus ']
    all_data_cqs_virus_merged = all_data_cqs_tissues.merge(
        all_data_cqs_virus[['HEX','FAM','log2-selection', 'pEK']], on='pEK', suffixes= ['_tissue','_virus'])
    all_data_cqs_virus_merged['organ'] = all_data_cqs_virus_merged['tissue'].apply(tissue_translator)
    all_data_cqs_virus_merged[
        'selection_virus_tissue'] = all_data_cqs_virus_merged[
        'log2-selection_virus'] - all_data_cqs_virus_merged['log2-selection_tissue']

    
    
    return mutant_info_df.merge(all_data_cqs_virus_merged.reset_index(), on='pEK').drop_duplicates()

# df = construct_validation_qPCR_df(meta_df_path_1, meta_df_path_2, data_df_path_1, data_df_path_2,mutant_info_df)

def make_all_frame_lookup(lookup_df,load=True, file_loc = 'frameshift_lookup.csv'):
    if load:
        frame_shift_lookup_df =  pd.read_csv(os.path.join(META_DIR,file_loc))
#         return frame_shift_lookup_df
        return LOOKUP_DF.merge(frame_shift_lookup_df.drop_duplicates(), on=['aa', 'abs_pos', 'codon'], how='left')
    
    frameshift_obj = []
    codon_table =  lookup_df[['aa', 'codon']].drop
    lookup_wt = lookup_df.query('is_wt_codon == 1')
    for idx, row in lookup_df.iterrows():
        if row['abs_pos'] == 1 or row['abs_pos'] == 1.5 or row['abs_pos'] >= 735:
            continue
        current_pos = row['abs_pos']

        if row['lib_type'] == 'del':continue
        if row['abs_pos'] % 1 != 0:
            position_value_changer = .5
            wt_codon = lookup_wt.query('abs_pos == (@current_pos + @position_value_changer)')['codon'].values[0]
        else:
            position_value_changer = 1
            wt_codon = lookup_wt.query('abs_pos == @current_pos')['codon'].values[0]
            
        prev_pos = row['abs_pos'] - position_value_changer
        next_pos = row['abs_pos'] + position_value_changer
        wt_next_codon = lookup_wt.query('abs_pos == @next_pos & is_wt_codon ==1')['codon'].values[0]
        wt_prev_codon = lookup_wt.query('abs_pos == @prev_pos & is_wt_codon ==1')['codon'].values[0]
        
        maap_codon_swap_first = '%s%s' % (wt_prev_codon[1:], row['codon'][0] )
        maap_codon_swap_second = '%s%s' % (row['codon'][1:], wt_next_codon[0])
        maap_aa_swap_first = str(Seq(maap_codon_swap_first).translate())
        maap_aa_swap_second = str(Seq(maap_codon_swap_second).translate())
        maap_wt_codon_first = '%s%s' % (wt_prev_codon[1:], wt_codon[0])
        maap_wt_codon_second = '%s%s' % (wt_codon[1:], wt_next_codon[0])
        maap_wt_aa_first = str(Seq(maap_wt_codon_first).translate())
        maap_wt_aa_second = str(Seq(maap_wt_codon_second).translate())
 
        frame3_codon_swap_first = '%s%s' % (wt_prev_codon[2], row['codon'][:2] )
        frame3_codon_swap_second = '%s%s' % (row['codon'][2], wt_next_codon[:2] )
        frame3_aa_swap_first = str(Seq(frame3_codon_swap_first).translate())
        frame3_aa_swap_second =  str(Seq(frame3_codon_swap_second).translate())
        frame3_wt_codon_first =  '%s%s' % (wt_prev_codon[2], wt_codon[:2] )
        frame3_wt_codon_second = '%s%s' % (wt_codon[2], wt_next_codon[:2] )
        frame3_wt_aa_first = str(Seq(frame3_wt_codon_first).translate())
        frame3_wt_aa_second = str(Seq(frame3_wt_codon_second).translate())

        frameshift_obj.append({'abs_pos':row['abs_pos'],
                               'aa':row['aa'],
                               'codon':row['codon'],
                               'maap_codon_swap_first':maap_codon_swap_first,
                               'maap_codon_swap_second':maap_codon_swap_second,
                               'maap_aa_swap_first':maap_aa_swap_first,
                               'maap_aa_swap_second': maap_aa_swap_second, 
                               'maap_wt_codon_first':maap_wt_codon_first,
                               'maap_wt_codon_second':maap_wt_codon_second,
                               'maap_wt_aa_first':maap_wt_aa_first,
                               'maap_wt_aa_second':maap_wt_aa_second,
                               'frame3_codon_swap_first':frame3_codon_swap_first,
                               'frame3_codon_swap_second':frame3_codon_swap_second,
                               'frame3_aa_swap_first':frame3_aa_swap_first,
                               'frame3_aa_swap_second':frame3_aa_swap_second,
                               'frame3_wt_codon_first':frame3_wt_codon_first,
                               'frame3_wt_codon_second':frame3_wt_codon_second,
                               'frame3_wt_aa_first':frame3_wt_aa_first,
                               'frame3_wt_aa_second':frame3_wt_aa_second})     
    df = pd.DataFrame(frameshift_obj)
    df.to_csv(file_loc, index=False)
    frame_shift_lookup_df =  pd.DataFrame(frameshift_obj)
    return LOOKUP_DF.merge(frame_shift_lookup_df.drop_duplicates(), on=['aa', 'abs_pos', 'codon'], how='left')
# frame_shift_lookup_df = make_all_frame_lookup(LOOKUP_DF)


# In[3]:


def plot_heatmap(selection_matrix,
                 lib_type='sub', 
                 promoter='CMV',
                 lib_num='0', 
                 save_dir=FIGURES_DIR, 
                 range_in=None,
                 fig_dimensions=[8,2],
                 dot_size=1,
                 text_size=7,
                 tick_size=.5,
                 plot_white=False,
                 line_width = .5,
                 min_max = (-5,5),
                 return_df = False,
                cmap='RdBu_r',
                save=False,
                plt_cbar=False,
                save_name=None):
    
    selection_matrix = selection_matrix.apply(np.log2)
    level ='aa-codon' if 'aa-codon' in selection_matrix.index.names else 'aa'
    order = DESIRED_AA_COD_ORD if level is 'aa-codon' else DESIRED_AA_ORD
    is_wt = 'is_wt_codon' if level == 'aa-codon' else 'is_wt_aa'
    
    if range_in is not None:
        selection_matrix = selection_matrix.loc[range_in]
    if level == 'aa-codon':
        df_heatmap_ready = selection_matrix.xs(
            lib_type, level='lib_type',drop_level=False ).xs((promoter, lib_num),level=[0,1], axis=1).reset_index(
            level=is_wt, drop=True).unstack(0).reset_index(
                level=[0,1],drop=True).reindex(DESIRED_AA_COD_ORD,level='aa-codon')
        df_heatmap_ready.index = df_heatmap_ready.index.droplevel(1)
    else:
        df_heatmap_ready = selection_matrix.xs(lib_type, level='lib_type',drop_level=False ).xs(
            (promoter, str(lib_num)),level=[0,1], axis=1).groupby(level=['aa', 'lib_type','abs_pos']).mean().unstack(
            'abs_pos').reset_index(level=1, drop=1).reindex(DESIRED_AA_ORD)
    
    if plot_white:
        df_heatmap_ready[:] = 0
    
    if range_in is not None:
        wt_plot_use = WT_PLOTTING_DF[WT_PLOTTING_DF['abs_pos'].isin(range_in)]
    else:
        wt_plot_use = WT_PLOTTING_DF
   
    wt_plot_use['aa-codon'] = wt_plot_use['aa'] + '-' + wt_plot_use['codon']
    mat_wt = wt_plot_use[wt_plot_use['wt']==1].pivot(level, 'abs_pos',values='wt').reindex(order)
    
    if level == 'aa-codon':
        line_splits=[]
        for idx,letter in enumerate(DESIRED_AA_COD_ORD):
            aa = letter.split('-')[0]
            aa_previous = DESIRED_AA_COD_ORD[idx-1].split('-')[0]
            if idx ==0: continue 
            if aa == aa_previous: 
                continue
#                 print aa, aa_previous
            if aa != aa_previous: 
                line_splits.append(idx)
    
    fig_length = 15 if level is 'aa' else 30
    if fig_dimensions:
        fig_params = fig_dimensions
    else:
        fig_params = [120, fig_length] 
    f,(ax1,ax2) = plt.subplots(
        1,2, sharex='col', figsize=fig_dimensions,gridspec_kw = {'width_ratios' : [100,.01],
                                                          'wspace':0.000025, 
                                                          'hspace':0.00005})
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
#     ax1.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylim([-64,0])
    ax2.set_xlim([-.5,.5])
#     sns.set(font_scale=.75)
    
    kwargs = {'xticklabels':df_heatmap_ready.columns.get_level_values(2)}
    
    if plt_cbar:
        sns.heatmap(df_heatmap_ready, vmin=min_max[0], vmax=min_max[1], cmap=cmap, 
                    yticklabels=order,ax=ax1,**kwargs)
    else:
        sns.heatmap(df_heatmap_ready, vmin=min_max[0], vmax=min_max[1], cmap=cmap,
                yticklabels=order,ax=ax1,cbar=False,**kwargs )
    if range_in is None:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
#     else:
#         ax1.set_xticks(ax1.get_xticks()[::10])

    
    ax1.tick_params(labelsize=tick_size,length=.5,width=.1,pad=.01)
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_ylabel('')
    ax1.set_xlabel('')

    lm = mat_wt.fillna(0).as_matrix()
    x,y = np.nonzero(lm.T)
    ax1.scatter(x+.4,y+.5,color='black', s=dot_size)
    if level is 'aa': sns.set(font_scale=.1)
    if level == 'aa-codon':
#         print 'level is aa-codon'
        for line_pos in line_splits:
            ax1.plot([0,735],[line_pos,line_pos],linewidth=line_width,color='black')
        for line_pos,aa in zip(line_splits,DESIRED_AA_ORD):
            position = -line_pos +1
            if aa == 'C': position += 1
            if aa == 'G': position +=1 if lib_type is 'sub' else 0
            if aa == 'M': position += -1 if lib_type is 'sub' else -2
            if aa == 'F': position += -.5 if lib_type is 'sub' else -1.5
            if aa == 'W': position += -1 if lib_type is 'sub' else -2
            if aa == 'Y': position += -.5 if lib_type is 'sub' else -1.5
            if aa == 'E': position += -.5 if lib_type is 'sub' else -2
            if aa == 'D': position += -.5 if lib_type is 'sub' else -2
            if aa == 'Q': position += -.5 if lib_type is 'sub' else -2
            if aa == 'N': position += -.5 if lib_type is 'sub' else -2
            if aa == 'H': position += -.5 if lib_type is 'sub' else -2.5
            if aa == 'C': position += - 1.5 if lib_type is 'sub' else -3.5
            if aa == 'R': position += + 2 if lib_type is 'sub' else 0
            if aa == 'K': position += - .5 if lib_type is 'sub' else -3
            if aa == 'S': position += + 2 if lib_type is 'sub' else 0
            if aa == 'T': position += + .5 if lib_type is 'sub' else -2
            if aa == 'P': position += + .5 if lib_type is 'sub' else -3
            if plt_cbar is False:
                ax2.text(0,position,aa,size=text_size,family ='monospace')    
    if save:
        if save_dir:
            if save_name:
                save_fig(f, save_name)
            else:
                save_name = os.path.join(save_dir, '%s_%s_%s_%s.pdf'  % (promoter, lib_type, level,lib_num))
                save_fig(f, save_name)
    if return_df:
        return {'df_heatmap' : df_heatmap_ready, 'mat_wt' :mat_wt}
    
def load_axis_contacts(contacts_df = pd.read_csv('../data/meta/viperdb_contatcs..csv')):
    contacts_df = contacts_df.drop(649)
    contacts_df['Residue1_pdb'] = contacts_df['Residue1-Residue2'].apply(lambda x: str(x).split('-')[0][3:])
    contacts_df['Residue2_pdb'] = contacts_df['Residue1-Residue2'].apply(lambda x: str(x).split('-')[1][3:])
    contacts_df['res1_vp'] = contacts_df['Residue1_pdb'].apply(int) +219-82
    contacts_df['res2_vp'] = contacts_df['Residue2_pdb'].apply(int) +219-82 # somehow this number is slightly different than pdb, but verified it is correct 
    return contacts_df


# In[ ]:




