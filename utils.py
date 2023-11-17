import collections
import torch
import gc
import pandas as pd
import matplotlib.pyplot as plt

def get_da(row):
    tup = eval(row['dialog_act'])
    return tup[0][:2] if tup else 'none'

def get_conv_mean(df, style):
    df[f'{style}_mean'] = df.loc[:,['conv_id',style]].groupby(['conv_id'])[style].transform('mean')
    return df

def get_lowest_subset(df, style, idx):
    return df.sort_values([f'{style}_mean', 'conv_id', 'turn_id']).reset_index(drop=True).loc[:idx,:]

def free_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

def stat_change_report(data, colname, details=False):
    import numpy as np
    arr = data[colname].to_numpy()
    tot = len(data[colname])
    pos, equ, neg = len(data[data[colname]>0]), len(data[data[colname]==0]), len(data[data[colname]<0])
    print(f"### STATS FOR {colname} ###")
    # print(f"Med: {np.median(arr)}")
    print(f"Avg: {np.mean(arr)}")
    print(f"Std: {np.std(arr)}")
    if details:
        print(f"Min: {np.min(arr)}")
        print(f"Max: {np.max(arr)}")
        print(f"N_pos: {pos}")
        print(f"N_equ: {equ}")
        print(f"N_neg: {neg}")
        print(f"%pos: {pos/tot}")
        print(f"%equ: {equ/tot}")
        print(f"%neg: {neg/tot}")
    print("")
    
def plot_print_count(old, new, colname, labels, figsize=(6.4,4.8)):
    data = pd.DataFrame()
    data[colname] = old[colname]
    data[f'new_{colname}'] = new[colname]
    count = {}
    for old_label in labels:
        count.update(collections.Counter(data.loc[data[colname]==old_label,f'new_{colname}'].tolist()))
        for key in labels:
            count[f'{old_label}_TO_{key}'] = count[key]
            del count[key]

    plt.rcParams["figure.figsize"] = figsize
    plt.bar(list(count.keys()), list(count.values()))
    plt.title(f'Count Plot for {colname}')
    plt.xlabel(f'Change in {colname}')
    plt.ylabel('Count')
    plt.show()
    
    return count