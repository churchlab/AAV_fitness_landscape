
# coding: utf-8

# # Process fastqs from geo into count files 
# NOTE: you will have to download the fastqs you wish to process from GEO/SRA and place them in `/data/fastqs/` for this script to function  
# If you wish to start with processed data, you can skip directly to notebooks in dir `/x02_analysis`

# In[1]:

import pandas as pd 
import re

from Bio import SeqIO 

import gzip
import os 
import re 
import sys
import time
import bz2


# ### read in sample info data from geo meta file 

# In[2]:

meta_df = pd.read_excel('../data/meta/GSE139657_meta.xls', 
                       skiprows=21,nrows=170 )


# In[3]:

meta_df.head()


# functions used to count barcodes and write counts 

# In[4]:

def tabulate(key, counts):
    # add 1 to count for this key, or add it to the dictionary
    if key in counts:
        counts[key] += 1
    else:
        counts[key] = 1

def write_counts(dict, outputfile):
    # write sequences and counts to a text file
#     print outputfile
    file_out = open(outputfile, 'w')
    file_out.write('barcode,count\n')
    for w in sorted(dict, key=dict.get, reverse=True):
        file_out.write('{seq},{num}\n'.format(seq=w, num=str(dict[w])))
    file_out.close() 
    


# ### using a row from the geo meta data, compute the counts for each barcode across all associated fastqs

# In[5]:

def count_ligated_bars(output_path = '../data/',
                       count_dir_name = 'counts',
                       log_out_name = 'count_log_file',
                       meta_df_row = None, 
                      limit=None):
    
    lib_file = os.path.join(output_path,'fastq',meta_df_row['raw file'])
    save_name = meta_df_row['title']
#     print (gen_id)
    output_counts_dir = os.path.join(output_path, count_dir_name)
    out_file_full = os.path.join(output_counts_dir , '%s.txt' % log_out_name)
#     if from_top:
#         if os.path.isfile(out_file_full):
#             os.remove(out_file_full)
    if os.path.isfile(out_file_full):
        log = open(out_file_full, 'a')
    else:
        log = open(out_file_full, 'w')
        log.write('reads\tbarcodes\tbarcodes_per_read\tlib_name\n')
    lig_dict ={}
    bar_count = 0
    read_count = 0 
#     print 'counting lib %s' % lib_file
    lig_dict = {}
 
    full_file_list = []
    for lane in ['L001', 'L002', 'L003','L004']:
        for read_direction in ['R1','R2']:
            lane_sub = re.sub('L001', lane, lib_file)
            full_file_name = re.sub('R1', read_direction,lane_sub )
            print (full_file_name)
            if os.path.isfile(full_file_name) == False:
                print ("file %s does not exist" % full_file_name)
                continue
            full_file_list.append(full_file_name)
            if 'gz' in full_file_name:
                handle = gzip.open(full_file_name)
            if 'bz' in full_file_name:
                print ("bz file")
                handle = bz2.open(full_file_name, 'rt')

            for idx, read in enumerate(SeqIO.parse(handle,'fastq')):
                read_count += 1 
                if limit:
                    if idx > limit : break 
                if 'R1' in read_direction:
                    seq = read.seq
                else:
                    seq = read.seq.reverse_complement()
    #             print seq # to debug
                middle_expr = '(?<=CCAC)([ACTG]{20})(?=CCAC)'
                matches = re.findall(middle_expr, str(seq))
                if matches:
                    for match in matches:
                        bar_count += 1
                        tabulate(match, lig_dict)

    print ("writing counts for %s" % save_name)
    assert len(full_file_list) == 8 , full_file_list
    log.write('%s\t%s\t%s\t%s\n' % (read_count,bar_count, (bar_count/float(read_count)), save_name))
    write_counts(lig_dict, os.path.join(output_counts_dir, '%s.csv' % (save_name)))
    log.close()


# example counting the barcodes for one of the sample

# In[6]:

count_ligated_bars(meta_df_row = meta_df.iloc[2])


# ### merge the barcode counts onto the meta-data table

# In[11]:

barcode_to_mutant_table = pd.read_csv('../data/meta/AAV2scan_chip_lookup_table.txt')
def assemble_dataframe(meta_df_row=meta_df.iloc[2], mutant_info_df = barcode_to_mutant_table ,save=False):
    filename = meta_df_row['title']
    count_file = os.path.join('../data','counts','%s.csv' % filename )
    count_df = pd.read_csv(count_file)
    count_with_mutant_info_df = barcode_to_mutant_table.merge(count_df, on='barcode',how='left')
    if save:
        count_with_mutant_info_df.to_csv(os.path.join('../data/counts', "%s_dataframe.csv" % filename))
    return count_with_mutant_info_df


# example for counts above

# In[12]:

count_data_with_mutant_info_df = assemble_dataframe()
count_data_with_mutant_info_df.head()


# In[ ]:



