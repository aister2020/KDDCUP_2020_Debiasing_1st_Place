#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from constants import *
import numpy as np
import pandas as pd
import ndcg_tools
import utils
cur_stage = CUR_STAGE
sum_mode = 'nosum' 

def gen_data(df, df_stage, mode):
    answer_source = utils.load_pickle( answer_source_path.format( mode, cur_stage, sum_mode ) )
    i2i_w02_recall_source = utils.load_pickle( i2i_w02_recall_scoure_path.format( mode, cur_stage, sum_mode ) )
    b2b_recall_source = utils.load_pickle( b2b_recall_scoure_path.format( mode, cur_stage, sum_mode ) )
    #i2i2i_recall_source = utils.load_pickle( i2i2i_recall_scoure_path.format( mode, cur_stage, sum_mode ) )
    i2i2i_new_recall_source = utils.load_pickle( i2i2i_new_recall_scoure_path.format( mode, cur_stage, sum_mode ) )
    i2i_w10_recall_source = utils.load_pickle( i2i_w10_recall_scoure_path.format( mode, cur_stage, sum_mode ) )

    recall_sources = [ i2i_w10_recall_source, b2b_recall_source, i2i2i_new_recall_source ] 
    recall_source_names = ['i2i_w10','b2b','i2i2i_new']  
    used_recall_source = '-'.join( recall_source_names ) 
    used_recall_source = used_recall_source+'-'+sum_mode
    print( f'Recall Source Use {used_recall_source} mode: {mode}')
    vals = df_stage[ ['user_id','stage'] ].values
    user2stage = {}
    for i in range( len(vals) ):
        user2stage[ vals[i][0] ] = vals[i][1]
    
    user2index = {}
    tdfs = []
    for sta in range(cur_stage+1):
        tdf = df_stage[ df_stage['stage']==sta ]['user_id'].values
        for i in range(tdf.shape[0]):    
            user2index[ tdf[i] ] = i 
    
    left_items_list = []
    left_times_list = []
    right_items_list = []
    right_times_list = []
    user_list = []
    time_list = []
    item_list = []
    sim_weight_list = []
    loc_weight_list = []
    time_weight_list = []
    rank_weight_list = []
    road_item_list = []
    road_item_loc_list = [] 
    road_item_time_list = []
    query_item_loc_list = []
    query_item_time_list = []
    recall_type_list = []
    stage_list = []
    label_list = []

    for user,group in df.groupby('user_id'):
        items = group['item_id'].values
        times = group['time'].values
        index = group['index'].values
        
        for i in range(len(items)):
            if index[i].endswith(mode):
                left_items = []
                left_times = []
                right_items = []
                right_times = []
                for k in range(i-1,-1,-1):
                    if not index[k].endswith(mode):
                        left_items.append(items[k])
                        left_times.append(times[k])
                
                for k in range(i+1,len(items)):
                    if not index[k].endswith(mode):
                        right_items.append(items[k])
                        right_times.append(times[k])
                
                # check
                if mode == 'valid':
                    if items[i]!=answer_source[user2stage[user]][user2index[user]][0]:
                        print(items[i],' ',answer_source[user2stage[user]][user2index[user]][0],' ',user2index[user])
                        print('召回出来的数据对应的pos数据对应不上。')
                        print(1/0)
                
                for idx,recall_source in enumerate(recall_sources):
                    recall = recall_source[user2stage[user]][user2index[user]]
                    '''
                    item: recall[j][0]
                    sim_weight: recall[j][1]
                    loc_weight: recall[j][2]
                    time_weight: recall[j][3]
                    rank_weight: recall[j][4]
                    road_item: recall[j][5]
                    road_item_loc: recall[j][6]
                    road_item_time: recall[j][7]
                    query_item_loc: recall[j][8]
                    query_item_time: recall[j][9]
                    '''
                    for j in range( len(recall) ):
                        user_list.append(user)
                        item_list.append(recall[j][0])
                        if recall[j][0] == items[i]:
                            label_list.append(1) 
                        else:
                            label_list.append(0)
                        sim_weight_list.append(recall[j][1])
                        loc_weight_list.append(recall[j][2])
                        time_weight_list.append(recall[j][3])
                        rank_weight_list.append(recall[j][4])
                        road_item_list.append(recall[j][5])
                        road_item_loc_list.append(recall[j][6])
                        road_item_time_list.append(recall[j][7])
                        query_item_loc_list.append(recall[j][8])
                        query_item_time_list.append(recall[j][9])
                        recall_type_list.append( idx )
                        stage_list.append( user2stage[user] )

                        left_items_list.append(left_items)
                        right_items_list.append(right_items)
                        left_times_list.append(left_times)
                        right_times_list.append(right_times)
                        time_list.append(times[i])
                    
    data = {}
    data['left_items_list'] = left_items_list
    data['right_items_list'] = right_items_list
    data['left_times_list'] = left_times_list
    data['right_times_list'] = right_times_list
    data['user'] = user_list
    data['time'] = time_list
    data['item'] = item_list
    data['sim_weight'] = sim_weight_list
    data['loc_weight'] = loc_weight_list
    data['time_weight'] = time_weight_list
    data['rank_weight'] = rank_weight_list
    data['road_item'] = road_item_list
    data['road_item_loc'] = road_item_loc_list
    data['road_item_time'] = road_item_time_list
    data['query_item_loc'] = query_item_loc_list
    data['query_item_time'] = query_item_time_list
    data['recall_type'] = recall_type_list
    data['stage'] = stage_list
    data['label'] = label_list
    data = pd.DataFrame(data)

    path = lgb_base_pkl.format( used_recall_source, mode, cur_stage )
    save_dir = '/'.join( path.split('/')[:-1] )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data.to_pickle( path )
    print(data.shape)
    

if __name__ == '__main__':
    seed = SEED
    neg = NEG
    mode = cur_mode
   
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
        df_valid = utils.load_pickle(all_valid_data_path.format(cur_stage))
        df_stage = utils.load_pickle(all_valid_stage_data_path.format(cur_stage))
        df = pd.concat([df_train,df_valid])
        df = df.sort_values(['user_id','time'])
    else:
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
        df_test = utils.load_pickle(online_all_test_data_path.format(cur_stage))
        df_stage = utils.load_pickle(online_all_test_data_path.format(cur_stage))
        df = pd.concat([df_train,df_test])
        df = df.sort_values(['user_id','time'])

    gen_data(df, df_stage, mode)
