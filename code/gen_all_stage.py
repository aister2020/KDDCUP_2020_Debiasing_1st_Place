# -*- coding: utf-8 -*-

from constants import *
import pandas as pd
import copy
import sys
import utils
from collections import defaultdict
import numpy as np

using_last_num = 'one'
online = 'offline'
if len(sys.argv)>0:
    online = sys.argv[1]
    using_last_num = sys.argv[2]
print('online: ', online, ' using_last_num: ', using_last_num)

def set_index(df_train, df_test, df_qtest):
    df_train['index'] = df_train.index
    df_train['index'] = df_train['index'].astype('str') + '_train'
    df_test['index'] = df_test.index
    df_test['index'] = df_test['index'].astype('str') + '_test_train'
    df_qtest['index'] = df_qtest.index
    df_qtest['index'] = df_qtest['index'].astype('str') + '_test'
   
def gen_stage_data(stage):
    
    dfs_train = []
    dfs_valid = []
    dfs_qtest = []
    
    for i in range(stage+1):
        df_train = pd.read_csv(get_train_item_click_file(i),names=['user_id','item_id','time'])
        df_test = pd.read_csv(get_test_item_click_file(i),names=['user_id','item_id','time'])
        
        print('before drop len',len(df_train))
        
        df_train = df_train.drop_duplicates(['user_id','item_id','time'])
        
        print('after drop len',len(df_train))
        
        print('before drop len',len(df_test))
        df_test = df_test.drop_duplicates(['user_id','item_id','time'])
        print('after drop len',len(df_test))
        
        
        df_train = df_train.sort_values(['user_id','time'])
        df_test = df_test.sort_values(['user_id','time'])
        if using_last_num == 'one':
            df_valid = df_test.groupby('user_id').last().reset_index()
        else:
            df_valid = df_test.groupby('user_id').tail(2).groupby('user_id').head(1).reset_index(drop=True)

        dfs_train.append(df_train)
        dfs_train.append(df_test)
        dfs_valid.append(df_valid)
        
        df_qtest = pd.read_csv(get_test_qitem_file(i),names=['user_id','time'])
        dfs_qtest.append(df_qtest)
    
    
    df_train = pd.concat(dfs_train)    
    
    print('before drop train len',len(df_train))
    df_train = df_train.drop_duplicates(['user_id','item_id','time'])
    print('after drop train len',len(df_train))

    
    df_valid = pd.concat(dfs_valid)
    print('before drop valid len',len(df_valid))
    df_valid = df_valid.drop_duplicates(['user_id','item_id','time'])
    print('after drop valid len',len(df_valid))
    
    
    for i,cur_df_qtest in enumerate(dfs_qtest):
        cur_df_qtest['stage'] = i
        dfs_qtest[i] = cur_df_qtest
    
    df_qtest = pd.concat(dfs_qtest)
    print('before drop qtest len',len(df_qtest))
    df_qtest = df_qtest.drop_duplicates(['user_id','time'])
    print('after drop qtest len',len(df_qtest))
    
    for i,cur_df_train in enumerate(dfs_train):
        cur_df_train['stage'] = int(i/2)
        dfs_train[i] = cur_df_train
        
    df_train_stage = pd.concat(dfs_train)
    print('before drop train_stage len',len(df_train_stage))
    df_train_stage = df_train_stage.drop_duplicates(['user_id','item_id','time','stage'])
    print('after drop train_stage len',len(df_train_stage))
    
    
    for i,cur_df_valid in enumerate(dfs_valid):
        cur_df_valid['stage'] = i
        dfs_valid[i] = cur_df_valid
        
    df_valid_stage = pd.concat(dfs_valid)
    print('before drop valid_stage len',len(df_valid_stage))
    df_valid_stage = df_valid_stage.drop_duplicates(['user_id','item_id','time','stage'])
    print('after drop valid_stage len',len(df_valid_stage))
    
    
    
    
    print('before drop train for valid len',len(df_train))
    df_train['drop_index'] = np.arange(len(df_train))
    # drop_index = df_train.merge(df_valid,on=['user_id','item_id','time'])['drop_index']
    drop_index = df_train.merge(df_valid,on=['user_id','item_id'])['drop_index']
    df_train = df_train[~df_train['drop_index'].isin(drop_index)]
    df_train = df_train.drop('drop_index',axis=1)
    print('before drop train for valid len',len(df_train))
    
    df_train = df_train.sort_values(['user_id','time']).reset_index(drop=True)
    df_train['time'] = (df_train['time'] - 0.98)*100
    df_train['index'] = df_train.index.astype('str') + '_train'
    
    
    print('before drop train_stage for valid len',len(df_train_stage))
    
    df_train_stage['drop_index'] = np.arange(len(df_train_stage))
    # drop_index = df_train_stage.merge(df_valid,on=['user_id','item_id','time'])['drop_index']
    drop_index = df_train_stage.merge(df_valid,on=['user_id','item_id'])['drop_index']
    df_train_stage = df_train_stage[~df_train_stage['drop_index'].isin(drop_index)]
    df_train_stage = df_train_stage.drop('drop_index',axis=1)
    
    print('after drop train_stage for valid len',len(df_train_stage))
    
    
    df_train_stage = df_train_stage.sort_values(['stage','user_id','time']).reset_index(drop=True)
    df_train_stage['time'] = (df_train_stage['time'] - 0.98)*100
    df_train_stage['index'] = df_train_stage.index.astype('str') + '_train_stage'
    
    
    
    df_valid = df_valid.sort_values(['user_id','time']).reset_index(drop=True)
    df_valid['time'] = (df_valid['time'] - 0.98)*100
    df_valid['index'] = df_valid.index.astype('str') + '_valid'
    
    
    df_qtest = df_qtest.sort_values(['user_id','time']).reset_index(drop=True)
    df_qtest['time'] = (df_qtest['time'] - 0.98)*100
    df_qtest['index'] = df_qtest.index.astype('str') + '_test'
    
    
    
    
    df_valid_stage = df_valid_stage.sort_values(['stage','user_id','time']).reset_index(drop=True)
    df_valid_stage['time'] = (df_valid_stage['time'] - 0.98)*100
    df_valid_stage['index'] = df_valid_stage.index.astype('str') + '_valid_stage'
    
    
    utils.dump_pickle(df_train, all_train_data_path.format(stage))
    utils.dump_pickle(df_train_stage, all_train_stage_data_path.format(stage))
    utils.dump_pickle(df_valid, all_valid_data_path.format(stage))
    utils.dump_pickle(df_valid_stage, all_valid_stage_data_path.format(stage))
    utils.dump_pickle(df_qtest, all_test_data_path.format(stage))    

def gen_stage_data_online(stage):
    dfs_train = []
    dfs_qtest = []
    
    for i in range(stage+1):
        df_train = pd.read_csv(get_train_item_click_file(i),names=['user_id','item_id','time'])
        df_test = pd.read_csv(get_test_item_click_file(i),names=['user_id','item_id','time'])
        
        print('before drop len',len(df_train))
        
        df_train = df_train.drop_duplicates(['user_id','item_id','time'])
        
        print('after drop len',len(df_train))
        
        print('before drop len',len(df_test))
        df_test = df_test.drop_duplicates(['user_id','item_id','time'])
        print('after drop len',len(df_test))
        
        
        df_train = df_train.sort_values(['user_id','time'])
        df_test = df_test.sort_values(['user_id','time'])
        
        dfs_train.append(df_train)
        dfs_train.append(df_test)
        
        df_qtest = pd.read_csv(get_test_qitem_file(i),names=['user_id','time'])
        dfs_qtest.append(df_qtest)
    
    
    df_train = pd.concat(dfs_train)    
    
    print('before drop train len',len(df_train))
    df_train = df_train.drop_duplicates(['user_id','item_id','time'])
    print('after drop train len',len(df_train))
    
    for i,cur_df_qtest in enumerate(dfs_qtest):
        cur_df_qtest['stage'] = i
        dfs_qtest[i] = cur_df_qtest
    
    df_qtest = pd.concat(dfs_qtest)
    print('before drop qtest len',len(df_qtest))
    df_qtest = df_qtest.drop_duplicates(['user_id','time'])
    print('after drop qtest len',len(df_qtest))
    
    for i,cur_df_train in enumerate(dfs_train):
        cur_df_train['stage'] = int(i/2)
        dfs_train[i] = cur_df_train
        
    df_train_stage = pd.concat(dfs_train)
    print('before drop train_stage len',len(df_train_stage))
    df_train_stage = df_train_stage.drop_duplicates(['user_id','item_id','time','stage'])
    print('after drop train_stage len',len(df_train_stage))
    
    
    df_train = df_train.sort_values(['user_id','time']).reset_index(drop=True)
    df_train['time'] = (df_train['time'] - 0.98)*100
    df_train['index'] = df_train.index.astype('str') + '_train'
    
    
    df_train_stage = df_train_stage.sort_values(['stage','user_id','time']).reset_index(drop=True)
    df_train_stage['time'] = (df_train_stage['time'] - 0.98)*100
    df_train_stage['index'] = df_train_stage.index.astype('str') + '_train_stage'
    
    
    df_qtest = df_qtest.sort_values(['user_id','time']).reset_index(drop=True)
    df_qtest['time'] = (df_qtest['time'] - 0.98)*100
    df_qtest['index'] = df_qtest.index.astype('str') + '_test'
    
    
    utils.dump_pickle(df_train, online_all_train_data_path.format(stage))
    utils.dump_pickle(df_train_stage, online_all_train_stage_data_path.format(stage))
    utils.dump_pickle(df_qtest, online_all_test_data_path.format(stage))    

def gen_item_degree(stage):
    phase_item_deg = {}
    item_deg = defaultdict(lambda: 0)
    for phase_id in range(stage+1):
        with open(get_train_item_click_file(phase_id)) as fin:
            for line in fin:
                user_id, item_id, timestamp = line.split(',')
                user_id, item_id, timestamp = (
                    int(user_id), int(item_id), float(timestamp))
                item_deg[item_id] += 1
        with open(get_test_item_click_file(phase_id)) as fin:
            for line in fin:
                user_id, item_id, timestamp = line.split(',')
                user_id, item_id, timestamp = (
                    int(user_id), int(item_id), float(timestamp))
                item_deg[item_id] += 1
        
        phase_item_deg[phase_id] = dict(item_deg)
        
    
    utils.dump_pickle(dict(item_deg),full_item_degree_path.format(stage))
    utils.dump_pickle(phase_item_deg,phase_full_item_degree_path.format(stage))
        
if __name__ == "__main__":
    stage = CUR_STAGE    
    stage = int(stage)
    
    if online == "offline":
        gen_stage_data(stage)
    elif online == "online":
        gen_stage_data_online(stage)
    else:
        print('online or offline error')
        print(1/0)
    
    gen_item_degree(stage)