#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from constants import *
import numpy as np
import pandas as pd
import utils
import time
from collections import deque, defaultdict
from scipy.spatial.distance import cosine
from scipy import stats
import math

seed = SEED
cur_stage = CUR_STAGE
mode = cur_mode
#used_recall_source = 'i2i_w02-b2b-i2i2i'
#used_recall_source = 'i2i_w02-b2b-i2i2i-i2i_w10'
#used_recall_source = 'i2i_w02-b2b-i2i2i-i2i_w10-i2i2b'
used_recall_source = cur_used_recall_source
sum_mode = 'nosum'
used_recall_source = used_recall_source+'-'+sum_mode
print( f'Recall Source Use {used_recall_source}')


def feat_item_sum_mean_sim_weight_loc_weight_time_weight_rank_weight(data):
    df = data.copy()
    df = df[ ['user','item','sim_weight','loc_weight','time_weight','rank_weight','index'] ]
    feat = df[ ['index','user','item'] ]
    df = df.groupby( ['user','item'] )[ ['sim_weight','loc_weight','time_weight','rank_weight'] ].agg( ['sum','mean'] ).reset_index()
    cols = [ f'item_{j}_{i}' for i in ['sim_weight','loc_weight','time_weight','rank_weight'] for j in ['sum','mean'] ]
    df.columns = [ 'user','item' ]+ cols
    feat = pd.merge( feat, df, on=['user','item'], how='left')
    feat = feat[ cols ] 
    return feat


def feat_sum_sim_loc_time_weight(data):
    df = data.copy()
    df = df[ ['index','sim_weight','loc_weight','time_weight'] ]
    feat = df[ ['index'] ]
    feat['sum_sim_loc_time_weight'] = df['sim_weight'] + df['loc_weight'] + df['time_weight']
    feat = feat[ ['sum_sim_loc_time_weight'] ]
    return feat

def feat_road_item_text_cossim(data):
    df = data.copy()
    df = df[ ['index','road_item','item'] ]
    feat = df[ ['index'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    for k,v in item_feat.items():
        item_text[k] = v[0]
    
    def func(ss):
        item1 = ss['road_item']
        item2 = ss['item']
        if ( item1 in item_text ) and ( item2 in item_text ):
            item1_text = item_text[item1]
            item2_text = item_text[item2]
            c = np.dot( item1_text, item2_text )
            a = np.linalg.norm( item1_text )
            b = np.linalg.norm( item2_text )
            return c/(a*b+(1e-9))
        else:
            return np.nan
    feat['road_item_text_cossim'] = df[ ['road_item','item'] ].apply(func, axis=1)
    feat = feat[ ['road_item_text_cossim'] ]
    return feat

def feat_road_item_text_eulasim(data):
    df = data.copy()
    df = df[ ['index','road_item','item'] ]
    feat = df[ ['index'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    for k,v in item_feat.items():
        item_text[k] = v[0]
    
    def func(ss):
        item1 = ss['road_item']
        item2 = ss['item']
        if ( item1 in item_text ) and ( item2 in item_text ):
            item1_text = item_text[item1]
            item2_text = item_text[item2]
            a = np.linalg.norm( item1_text - item2_text )
            return a
        else:
            return np.nan
    feat['road_item_text_eulasim'] = df[ ['road_item','item'] ].apply(func, axis=1)
    feat = feat[ ['road_item_text_eulasim'] ]
    return feat

def feat_road_item_text_mansim(data):
    df = data.copy()
    df = df[ ['index','road_item','item'] ]
    feat = df[ ['index'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    for k,v in item_feat.items():
        item_text[k] = v[0]
    
    def func(ss):
        item1 = ss['road_item']
        item2 = ss['item']
        if ( item1 in item_text ) and ( item2 in item_text ):
            item1_text = item_text[item1]
            item2_text = item_text[item2]
            a = np.linalg.norm( item1_text - item2_text, ord=1 )
            return a
        else:
            return np.nan
    feat['road_item_text_mansim'] = df[ ['road_item','item'] ].apply(func, axis=1)
    feat = feat[ ['road_item_text_mansim'] ]
    return feat

def feat_road_item_image_cossim(data):
    df = data.copy()
    df = df[ ['index','road_item','item'] ]
    feat = df[ ['index'] ]
    item_image = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    for k,v in item_feat.items():
        item_image[k] = v[1]
    
    def func(ss):
        item1 = ss['road_item']
        item2 = ss['item']
        if ( item1 in item_image ) and ( item2 in item_image ):
            item1_image = item_image[item1]
            item2_image = item_image[item2]
            c = np.dot( item1_image, item2_image )
            a = np.linalg.norm( item1_image )
            b = np.linalg.norm( item2_image )
            return c/(a*b+(1e-9))
        else:
            return np.nan
    feat['road_item_image_cossim'] = df[ ['road_item','item'] ].apply(func, axis=1)
    feat = feat[ ['road_item_image_cossim'] ]
    return feat

def feat_road_item_image_eulasim(data):
    df = data.copy()
    df = df[ ['index','road_item','item'] ]
    feat = df[ ['index'] ]
    item_image = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    for k,v in item_feat.items():
        item_image[k] = v[1]
    
    def func(ss):
        item1 = ss['road_item']
        item2 = ss['item']
        if ( item1 in item_image ) and ( item2 in item_image ):
            item1_image = item_image[item1]
            item2_image = item_image[item2]
            a = np.linalg.norm( item1_image - item2_image )
            return a
        else:
            return np.nan
    feat['road_item_image_eulasim'] = df[ ['road_item','item'] ].apply(func, axis=1)
    feat = feat[ ['road_item_image_eulasim'] ]
    return feat

def feat_road_item_image_mansim(data):
    df = data.copy()
    df = df[ ['index','road_item','item'] ]
    feat = df[ ['index'] ]
    item_image = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    for k,v in item_feat.items():
        item_image[k] = v[0]
    
    def func(ss):
        item1 = ss['road_item']
        item2 = ss['item']
        if ( item1 in item_image ) and ( item2 in item_image ):
            item1_image = item_image[item1]
            item2_image = item_image[item2]
            a = np.linalg.norm( item1_image - item2_image, ord=1 )
            return a
        else:
            return np.nan
    feat['road_item_image_mansim'] = df[ ['road_item','item'] ].apply(func, axis=1)
    feat = feat[ ['road_item_image_mansim'] ]
    return feat

def feat_i2i_seq(data):  
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = set()
    for val in vals:
        new_keys.add( (val[0], val[1]) )

    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    i2i_sim_seq = {}

    st0 = time.time()
    tot = 0
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        if tot % 500 == 0:
            print( f'tot: {len(user_item_dict)}, now: {tot}' )
        tot += 1
        for loc1, item in enumerate(items):
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                if (item,relate_item) not in new_keys:
                    continue
                t1 = times[loc1]
                t2 = times[loc2]
                i2i_sim_seq.setdefault((item,relate_item), [])
                i2i_sim_seq[ (item,relate_item) ].append( (loc1, loc2, t1, t2, len(items) ) )


    st1 = time.time()
    print(st1-st0)
    return i2i_sim_seq 

def feat_i2i2i_seq(data):  
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = set()
    for val in vals:
        new_keys.add( (val[0], val[1]) )

    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    
    all_pair_num = 0
    sim_item_p2 = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item_p2.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item_p2[item].setdefault(relate_item, 0)
                
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (0.9**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2

                    sim_item_p2[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items))
                          
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (0.9**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                    sim_item_p2[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items)) 
    
    sim_item_p1 = {}
    for i, related_items in sim_item_p2.items():  
        sim_item_p1[i] = {}
        for j, cij in related_items.items():  
            sim_item_p1[i][j] = cij / (item_cnt[i] * item_cnt[j])
            sim_item_p2[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)
    

    print('all_pair_num',all_pair_num)
    for key in sim_item_p2.keys():
        t = sim_item_p2[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:50]:
            res[i[0]]=i[1]
        sim_item_p2[key] = res

    i2i2i_sim_seq = {}
    t1 = time.time()
    for idx,item1 in enumerate( sim_item_p2.keys() ):
        if idx%10000==0:
            t2 = time.time()
            print( f'use time {t2-t1} for 10000, now {idx} , tot {len(sim_item_p2.keys())}' )
            t1 = t2
        for item2 in sim_item_p2[item1].keys():
            if item2 == item1:
                continue
            for item3 in sim_item_p2[item2].keys():
                if item3 == item1 or item3 == item2:
                    continue
                if (item1,item3) not in new_keys:
                    continue
                i2i2i_sim_seq.setdefault((item1,item3), [])
                i2i2i_sim_seq[ (item1,item3) ].append( ( item2, sim_item_p2[item1][item2], sim_item_p2[item2][item3],
                                                        sim_item_p1[item1][item2], sim_item_p1[item2][item3] ) )

    return i2i2i_sim_seq 

def feat_i2i2b_seq(data):  
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = set()
    for val in vals:
        new_keys.add( (val[0], val[1]) )

    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    
    all_pair_num = 0
    sim_item_p2 = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item_p2.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item_p2[item].setdefault(relate_item, 0)
                
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (0.9**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2

                    sim_item_p2[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items))
                          
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (0.9**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                    sim_item_p2[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items)) 
    
    sim_item_p1 = {}
    for i, related_items in sim_item_p2.items():  
        sim_item_p1[i] = {}
        for j, cij in related_items.items():  
            sim_item_p1[i][j] = cij / (item_cnt[i] * item_cnt[j])
            sim_item_p2[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)
    

    print('all_pair_num',all_pair_num)
    for key in sim_item_p2.keys():
        t = sim_item_p2[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:100]:
            res[i[0]]=i[1]
        sim_item_p2[key] = res


    blend_sim = utils.load_sim(item_blend_sim_path)
    blend_score = {}
    
    for item in blend_sim:
        i = item[0]
        blend_score.setdefault(i,{})
        for j,cij in item[1][:100]:
            blend_score[i][j] = cij


    i2i2b_sim_seq = {}
    t1 = time.time()
    for idx,item1 in enumerate( sim_item_p2.keys() ):
        if idx%10000==0:
            t2 = time.time()
            print( f'use time {t2-t1} for 10000, now {idx} , tot {len(sim_item_p2.keys())}' )
            t1 = t2
        for item2 in sim_item_p2[item1].keys():
            if (item2 == item1) or (item2 not in blend_score.keys()):
                continue
            for item3 in blend_score[item2].keys():
                if item3 == item1 or item3 == item2:
                    continue
                if (item1,item3) not in new_keys:
                    continue
                i2i2b_sim_seq.setdefault((item1,item3), [])
                i2i2b_sim_seq[ (item1,item3) ].append( ( item2, sim_item_p2[item1][item2], blend_score[item2][item3],
                                                        sim_item_p1[item1][item2], blend_score[item2][item3] ) )

    return i2i2b_sim_seq 

def feat_i2i_sim(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1


    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    result = {}
    for key in new_keys:
        if key not in i2i_sim_seq.keys():
            result[key] = np.nan
            continue
        result[key] = 0.0
        records = i2i_sim_seq[key]
        if len(records)==0:
            print(key)
        for record in records:
            loc1, loc2, t1, t2, record_len = record
            if loc1-loc2>0:
                time_weight = (1 - (t1 - t2) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                    
                loc_diff = loc1-loc2-1
                loc_weight = (0.9**loc_diff)
                if loc_weight <= 0.2:
                    loc_weight = 0.2        
            else:
                time_weight = (1 - (t2 - t1) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                
                loc_diff =  loc2-loc1-1
                loc_weight =  (0.9**loc_diff)
                
                if loc_weight <= 0.2:
                    loc_weight = 0.2
                
            result[key] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len) 
    
    for key in new_keys:
        if np.isnan( result[key] ):
            continue
        result[key] = result[key] / ((item_cnt[key[0]] * item_cnt[key[1]]) ** 0.2)
    print('Finished getting result')
    feat['i2i_sim'] = feat['new_keys'].map(result)
    #import pdb
    #pdb.set_trace()
    #i2i_seq_feat = pd.concat( [feat,i2i_seq_feat], axis=1 )
    #i2i_seq_feat['itemAB'] = i2i_seq_feat['road_item'].astype('str') + '-' + i2i_seq_feat['item'].astype('str')
    feat = feat[ ['i2i_sim'] ]
    return feat

def feat_i2i_sim_abs_loc_weights_loc_base(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    loc_bases = [0.2,0.4,0.6,0.8,1.0]
    for loc_base in loc_bases:
        print(f'Starting {loc_base}')
        result = {}
        for key in new_keys:
            if key not in i2i_sim_seq.keys():
                result[key] = np.nan
                continue
            result[key] = 0.0
            records = i2i_sim_seq[key]
            if len(records)==0:
                print(key)
            for record in records:
                loc1, loc2, t1, t2, record_len = record
                if loc1-loc2>0:       
                    loc_diff = loc1-loc2-1
                    loc_weight = (loc_base**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2        
                else:
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (loc_base**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                result[key] += loc_weight

        feat['i2i_sim_abs_loc_weights_loc_base'+str(loc_base)] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = []
    for loc_base in loc_bases:
        cols.append( 'i2i_sim_abs_loc_weights_loc_base'+str(loc_base) )
    feat = feat[ cols ]
    return feat

def feat_i2i_sim_loc_weights_loc_base(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    loc_bases = [0.2,0.4,0.6,0.8,1.0]
    for loc_base in loc_bases:
        print(f'Starting {loc_base}')
        result = {}
        for key in new_keys:
            if key not in i2i_sim_seq.keys():
                result[key] = np.nan
                continue
            result[key] = 0.0
            records = i2i_sim_seq[key]
            if len(records)==0:
                print(key)
            for record in records:
                loc1, loc2, t1, t2, record_len = record
                loc_diff = loc1-loc2
                loc_weight = (loc_base**loc_diff)
                if abs(loc_weight) <= 0.2:
                    if loc_weight > 0:
                        loc_weight = 0.2 
                    else:
                        loc_weight = -0.2        
                result[key] += loc_weight

        feat['i2i_sim_loc_weights_loc_base'+str(loc_base)] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = []
    for loc_base in loc_bases:
        cols.append( 'i2i_sim_loc_weights_loc_base'+str(loc_base) )
    feat = feat[ cols ]
    return feat

def feat_i2i_sim_abs_time_weights(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    
    result = {}
    for key in new_keys:
        if key not in i2i_sim_seq.keys():
            result[key] = np.nan
            continue
        result[key] = 0.0
        records = i2i_sim_seq[key]
        if len(records)==0:
            print(key)
        for record in records:
            loc1, loc2, t1, t2, record_len = record
            if loc1-loc2>0:
                time_weight = (1 - (t1 - t2) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
            else:
                time_weight = (1 - (t2 - t1) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2

            result[key] += time_weight

    feat['i2i_sim_abs_time_weights'] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = [ 'i2i_sim_abs_time_weights' ]
    feat = feat[ cols ]
    return feat

def feat_i2i_sim_time_weights(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    
    result = {}
    for key in new_keys:
        if key not in i2i_sim_seq.keys():
            result[key] = np.nan
            continue
        result[key] = 0.0
        records = i2i_sim_seq[key]
        if len(records)==0:
            print(key)
        for record in records:
            loc1, loc2, t1, t2, record_len = record
            
            time_weight = (1 - (t1 - t2) * 100)
            if abs(time_weight)<=0.2:
                if time_weight > 0:
                    time_weight = 0.2
                else:
                    time_weight = -0.2
         
            result[key] += time_weight

    feat['i2i_sim_time_weights'] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = [ 'i2i_sim_time_weights' ]
    feat = feat[ cols ]
    return feat

def feat_i2i_cijs_abs_loc_weights_loc_base(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    loc_bases = [0.2,0.4,0.6,0.8,1.0]
    for loc_base in loc_bases:
        print(f'Starting {loc_base}')
        result = {}
        for key in new_keys:
            if key not in i2i_sim_seq.keys():
                result[key] = np.nan
                continue
            result[key] = 0.0
            records = i2i_sim_seq[key]
            if len(records)==0:
                print(key)
            for record in records:
                loc1, loc2, t1, t2, record_len = record
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (loc_base**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2        
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (loc_base**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                result[key] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len) 

        feat['i2i_cijs_abs_loc_weights_loc_base_'+str(loc_base)] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = []
    for loc_base in loc_bases:
        cols.append( 'i2i_cijs_abs_loc_weights_loc_base_'+str(loc_base) )
    feat = feat[ cols ]
    return feat

def feat_i2i_cijs_loc_weights_loc_base(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    loc_bases = [0.2,0.4,0.6,0.8,1.0]
    for loc_base in loc_bases:
        print(f'Starting {loc_base}')
        result = {}
        for key in new_keys:
            if key not in i2i_sim_seq.keys():
                result[key] = np.nan
                continue
            result[key] = 0.0
            records = i2i_sim_seq[key]
            if len(records)==0:
                print(key)
            for record in records:
                loc1, loc2, t1, t2, record_len = record
                time_weight = (1 - abs(t2 - t1) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                loc_diff =  abs(loc2-loc1)
                loc_weight =  (loc_base**loc_diff)
                if loc_weight <= 0.2:
                    loc_weight = 0.2
                if loc1-loc2>0:
                    result[key] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len) 
                else:
                    result[key] -= 1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len) 

        feat['i2i_cijs_loc_weights_loc_base_'+str(loc_base)] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = []
    for loc_base in loc_bases:
        cols.append( 'i2i_cijs_loc_weights_loc_base_'+str(loc_base) )
    feat = feat[ cols ]
    return feat

def feat_i2i_cijs_mean_abs_loc_weights_loc_base(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    loc_bases = [0.2,0.4,0.6,0.8,1.0]
    for loc_base in loc_bases:
        print(f'Starting {loc_base}')
        result = {}
        for key in new_keys:
            if key not in i2i_sim_seq.keys():
                result[key] = np.nan
                continue
            result[key] = 0.0
            records = i2i_sim_seq[key]
            if len(records)==0:
                print(key)
            for record in records:
                loc1, loc2, t1, t2, record_len = record
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (loc_base**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2        
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (loc_base**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                result[key] += ( 1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len) ) / len(records)

        feat['i2i_cijs_mean_abs_loc_weights_loc_base_'+str(loc_base)] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = []
    for loc_base in loc_bases:
        cols.append( 'i2i_cijs_mean_abs_loc_weights_loc_base_'+str(loc_base) )
    feat = feat[ cols ]
    return feat

def feat_i2i_bottom_itemcnt_sum_weight(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    #print('Loading i2i_sim_seq')
    #i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    #print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    weights = [0.2,0.4,0.6,0.8,1.0]
    for weight in weights:
        print(f'Starting {weight}')
        result = {}
        for key in new_keys:
            if (key[0] in item_cnt.keys()) and (key[1] in item_cnt.keys()):
                result[key] = ((item_cnt[key[0]] + item_cnt[key[1]]) ** weight)

        feat['i2i_bottom_itemcnt_sum_weight_'+str(weight)] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = []
    for weight in weights:
        cols.append( 'i2i_bottom_itemcnt_sum_weight_'+str(weight) )
    feat = feat[ cols ]
    return feat

def feat_i2i_bottom_itemcnt_multi_weight(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    #print('Loading i2i_sim_seq')
    #i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    #print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    weights = [0.2,0.4,0.6,0.8,1.0]
    for weight in weights:
        print(f'Starting {weight}')
        result = {}
        for key in new_keys:
            if (key[0] in item_cnt.keys()) and (key[1] in item_cnt.keys()):
                result[key] = ((item_cnt[key[0]] * item_cnt[key[1]]) ** weight)

        feat['i2i_bottom_itemcnt_multi_weight_'+str(weight)] = feat['new_keys'].map(result)
    print('Finished getting result')
    cols = []
    for weight in weights:
        cols.append( 'i2i_bottom_itemcnt_multi_weight_'+str(weight) )
    feat = feat[ cols ]
    return feat

def feat_b2b_sim(data):
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    blend_sim = utils.load_sim(item_blend_sim_path)
    b2b_sim = {}
    for item in blend_sim:
        i = item[0]
        b2b_sim.setdefault(i,{})
        for j,cij in item[1][:100]:
            b2b_sim[i][j] = cij

    vals = feat[ ['road_item','item'] ].values
    result = []
    for val in vals:
        item1 = val[0]
        item2 = val[1]
        if item1 in b2b_sim.keys():
            if item2 in b2b_sim[item1].keys():
                result.append( b2b_sim[ item1 ][ item2 ] )
            else:
                result.append( np.nan )
        else:
            result.append( np.nan )
    feat['b2b_sim'] = result
    feat = feat[ ['b2b_sim'] ]
    return feat

def feat_itemqa_loc_diff(data):
    df = data.copy()
    feat = df[ ['index','query_item_loc','road_item_loc'] ]
    feat['itemqa_loc_diff'] = feat['road_item_loc'] - feat['query_item_loc']
    def func(s):
        if s<0:
            return -s
        return s
    feat['abs_itemqa_loc_diff'] = feat['itemqa_loc_diff'].apply(func)
    feat = feat[ ['itemqa_loc_diff','abs_itemqa_loc_diff'] ]
    return feat

def feat_sim_three_weight(data):  
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    loc_weights = {}
    time_weights = {}
    record_weights = {}
    com_item_cnt = {}
    
    item_set = set()
    item_dict_set = {}
    st0 = time.time()
    
    for user, items in user_item_dict.items():
        for item in items:
            item_set.add(item)
            item_dict_set[item] = set()
    
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            loc_weights.setdefault(item, {})
            time_weights.setdefault(item, {})
            record_weights.setdefault(item, {})
            com_item_cnt.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                item_dict_set[ item ].add( relate_item )
                
                t1 = times[loc1]
                t2 = times[loc2]
                loc_weights[item].setdefault(relate_item, 0)
                time_weights[item].setdefault(relate_item, 0)
                record_weights[item].setdefault(relate_item, 0)
                com_item_cnt[item].setdefault(relate_item, 0)
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (0.9**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2

                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (0.9**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                loc_weights[item][relate_item] += loc_weight
                time_weights[item][relate_item] += time_weight
                record_weights[item][relate_item] += len(items)
                com_item_cnt[item][relate_item] += 1

    st1 = time.time()
    print(st1-st0)

    print('start')
    num = feat.shape[0]
    road_item = feat['road_item'].values
    t_item = feat['item'].values
    
    com_item_loc_weights_sum = np.zeros( num, dtype=float )
    com_item_time_weights_sum = np.zeros( num, dtype=float )
    com_item_record_weights_sum = np.zeros( num, dtype=float )
    t_com_item_cnt = np.zeros( num, dtype=float )
    for i in range(num):
        if road_item[i] in item_set:
            if t_item[i] in item_dict_set[ road_item[i] ]:
                com_item_loc_weights_sum[i] = loc_weights[ road_item[i] ][ t_item[i] ] 
                com_item_time_weights_sum[i] = time_weights[ road_item[i] ][ t_item[i] ] 
                com_item_record_weights_sum[i] = record_weights[ road_item[i] ][ t_item[i] ] 
                t_com_item_cnt[i] = com_item_cnt[ road_item[i] ][ t_item[i] ] 
            else:
                com_item_loc_weights_sum[i] = np.nan
                com_item_time_weights_sum[i] = np.nan
                com_item_record_weights_sum[i] = np.nan
                t_com_item_cnt[i] = np.nan
        else:
            com_item_loc_weights_sum[i] = np.nan
            com_item_time_weights_sum[i] = np.nan
            com_item_record_weights_sum[i] = np.nan
            t_com_item_cnt[i] = np.nan

    feat['com_item_loc_weights_sum'] = com_item_loc_weights_sum
    feat['com_item_time_weights_sum'] = com_item_time_weights_sum
    feat['com_item_record_weights_sum'] = com_item_record_weights_sum
    feat['com_item_cnt'] = t_com_item_cnt
    
    feat['com_item_loc_weights_mean'] = feat['com_item_loc_weights_sum'] / feat['com_item_cnt']
    feat['com_item_time_weights_mean'] = feat['com_item_time_weights_sum'] / feat['com_item_cnt']
    feat['com_item_record_weights_mean'] = feat['com_item_record_weights_sum'] / feat['com_item_cnt']

    feat = feat[ ['com_item_loc_weights_sum','com_item_time_weights_sum','com_item_record_weights_sum',
                  'com_item_loc_weights_mean','com_item_time_weights_mean','com_item_record_weights_mean' ] ]

    st2 = time.time()
    print(st2-st1)
    return feat

def feat_different_type_road_score_sum_mean(data):  
    df = data.copy()
    feat = df[ ['user','item','index','sim_weight','recall_type'] ]
    feat['i2i_score'] = feat['sim_weight']
    feat['blend_score'] = feat['sim_weight']
    feat['i2i2i_score'] = feat['sim_weight']
    feat.loc[ feat['recall_type']!=0 , 'i2i_score'] = np.nan
    feat.loc[ feat['recall_type']!=1 , 'blend_score'] = np.nan
    feat.loc[ feat['recall_type']!=2 , 'i2i2i_score'] = np.nan
    feat['user_item'] = feat['user'].astype('str') + '-' + feat['item'].astype('str')
    
    for col in ['i2i_score','blend_score','i2i2i_score']:
        df = feat[ ['user_item',col,'index'] ]
        df = df.groupby('user_item')[col].sum().reset_index()
        df[col+'_sum'] = df[col]
        df = df[ ['user_item',col+'_sum'] ]
        feat = pd.merge( feat, df, on='user_item', how='left')
        
        df = feat[ ['user_item',col,'index'] ]
        df = df.groupby('user_item')[col].mean().reset_index()
        df[col+'_mean'] = df[col]
        df = df[ ['user_item',col+'_mean'] ]
        feat = pd.merge( feat, df, on='user_item', how='left')
        
        
    feat = feat[ ['i2i_score','i2i_score_sum','i2i_score_mean',
                  'blend_score','blend_score_sum','blend_score_mean',
                  'i2i2i_score','i2i2i_score_sum','i2i2i_score_mean',] ]
    return feat

def feat_different_type_road_score_sum_mean_new(data):  
    df = data.copy()
    feat = df[ ['user','item','index','sim_weight','recall_type'] ]

    recall_source_names = ['i2i_w02','b2b','i2i2i','i2i_w10','i2i2b'] 
    recall_source_names = [ i+'_score' for i in recall_source_names ] 
    for idx,col in enumerate(recall_source_names):
        feat[col] = feat['sim_weight']
        feat.loc[ feat['recall_type']!=idx, col ] = np.nan

    for col in recall_source_names:
        df = feat[ ['user','item',col,'index'] ]
        df = df.groupby( ['user','item'] )[col].sum().reset_index()
        df[col+'_sum'] = df[col]
        df = df[ ['user','item',col+'_sum'] ]
        feat = pd.merge( feat, df, on=['user','item'], how='left')
        
        df = feat[ ['user','item',col,'index'] ]
        df = df.groupby( ['user','item'] )[col].mean().reset_index()
        df[col+'_mean'] = df[col]
        df = df[ ['user','item',col+'_mean'] ]
        feat = pd.merge( feat, df, on=['user','item'], how='left')
    feat_list = recall_source_names + [ col+'_sum' for col in recall_source_names ] + [ col+'_mean' for col in recall_source_names ]
    feat = feat[ feat_list ]
    return feat

def feat_sim_base(data):
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    sim_item = {}
    item_cnt = defaultdict(int)
    com_item_cnt = {}
    
    item_set = set()
    item_dict_set = {}
    
    st0 = time.time()
    for user, items in user_item_dict.items():
        for item in items:
            item_set.add(item)
            item_dict_set[item] = set()
            
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            com_item_cnt.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                item_dict_set[ item ].add( relate_item )
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                com_item_cnt[item].setdefault(relate_item, 0)
                
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (0.9**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2

                    sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items))
                          
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (0.9**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                    sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items)) 
                com_item_cnt[item][relate_item] += 1.0
    
    st1 = time.time()
    print(st1-st0)
    print('start')
    num = feat.shape[0]
    road_item = feat['road_item'].values
    t_item = feat['item'].values
    
    road_item_cnt = np.zeros( num, dtype=float )
    t_item_cnt = np.zeros( num, dtype=float )
    com_item_cij = np.zeros( num, dtype=float )
    t_com_item_cnt = np.zeros( num, dtype=float )

    for i in range(num):
        if road_item[i] in item_set:
            road_item_cnt[i] = item_cnt[ road_item[i] ]
            
            if t_item[i] in item_dict_set[ road_item[i] ]:
                com_item_cij[i] = sim_item[ road_item[i] ][ t_item[i] ] 
                t_com_item_cnt[i] = com_item_cnt[ road_item[i] ][ t_item[i] ] 
            else:
                com_item_cij[i] = np.nan
                t_com_item_cnt[i] = np.nan
        else:
            road_item_cnt[i] = np.nan
            com_item_cij[i] = np.nan
            t_com_item_cnt[i] = np.nan
        
        if t_item[i] in item_set:
            t_item_cnt[i] = item_cnt[ t_item[i] ]
        else:
            t_item_cnt[i] = np.nan

    
    feat['road_item_cnt'] = road_item_cnt
    feat['item_cnt'] = t_item_cnt
    feat['com_item_cij'] = com_item_cij
    feat['com_item_cnt'] = t_com_item_cnt

    feat = feat[ ['road_item_cnt','item_cnt','com_item_cij','com_item_cnt' ] ]
    
    st2 = time.time()
    print(st2-st1)
    return feat

def feat_u2i_abs_loc_weights_loc_base(data):
    df = data.copy()
    feat = df[ ['road_item','item','query_item_loc','query_item_time','road_item_loc','road_item_time'] ]
    vals = feat[ ['query_item_loc','road_item_loc'] ].values
    loc_bases = [0.1,0.3,0.5,0.7,0.9]
    for loc_base in loc_bases:
        result = []
        for val in vals:
            loc1 = val[0]
            loc2 = val[1]
            if loc2 >= loc1:
                loc_diff = loc2-loc1
            else:
                loc_diff = loc1-loc2-1
            loc_weight = loc_base**loc_diff
            if loc_weight<=0.1:
                loc_weight = 0.1
            result.append(loc_weight)
        feat['u2i_abs_loc_weights_loc_base_'+str(loc_base)] = result 
    cols = []
    for loc_base in loc_bases:
        cols.append( 'u2i_abs_loc_weights_loc_base_'+str(loc_base) )
    feat = feat[ cols ]
    return feat

def feat_u2i_loc_weights_loc_base(data):
    df = data.copy()
    feat = df[ ['road_item','item','query_item_loc','query_item_time','road_item_loc','road_item_time'] ]
    vals = feat[ ['query_item_loc','road_item_loc'] ].values
    loc_bases = [0.1,0.3,0.5,0.7,0.9]
    for loc_base in loc_bases:
        result = []
        for val in vals:
            loc1 = val[0]
            loc2 = val[1]
            if loc2 >= loc1:
                loc_diff = loc2-loc1
            else:
                loc_diff = loc1-loc2-1
            loc_weight = loc_base**loc_diff

            if abs(loc_weight)<=0.1:
                loc_weight = 0.1
            if loc2 < loc1:
                loc_weight = -loc_weight
            result.append(loc_weight)
        feat['u2i_loc_weights_loc_base_'+str(loc_base)] = result 
    cols = []
    for loc_base in loc_bases:
        cols.append( 'u2i_loc_weights_loc_base_'+str(loc_base) )
    feat = feat[ cols ]
    return feat

def feat_u2i_abs_time_weights(data):
    df = data.copy()
    feat = df[ ['road_item','item','query_item_loc','query_item_time','road_item_loc','road_item_time'] ]
    vals = feat[ ['query_item_time','road_item_time'] ].values
    result = []
    for val in vals:
        t1 = val[0]
        t2 = val[1]
        time_weight = (1 - abs( t1 - t2 ) * 100)
        if time_weight<=0.1:
            time_weight = 0.1
        result.append(time_weight)
    feat['u2i_abs_time_weights'] = result 
    cols = [ 'u2i_abs_time_weights' ]
    feat = feat[ cols ]
    return feat

def feat_u2i_time_weights(data):
    df = data.copy()
    feat = df[ ['road_item','item','query_item_loc','query_item_time','road_item_loc','road_item_time'] ]
    vals = feat[ ['query_item_time','road_item_time'] ].values
    result = []
    for val in vals:
        t1 = val[0]
        t2 = val[1]
        time_weight = (1 - abs( t1 - t2 ) * 100)
        if abs(time_weight)<=0.1:
            time_weight = 0.1
        if t1 > t2:
            time_weight = -time_weight
        result.append(time_weight)
    feat['u2i_time_weights'] = result 
    cols = [ 'u2i_time_weights' ]
    feat = feat[ cols ]
    return feat

def feat_automl_cate_count(data):
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    feat['road_item-item'] = feat['road_item'].astype('str') + '-' + feat['item'].astype('str')
    cate_list = [ 'road_item','item','road_item-item' ]
    cols = []
    for cate in cate_list:
        feat[cate+'_count'] = feat[ cate ].map( feat[ cate ].value_counts() )
        cols.append( cate+'_count' )    
    feat = feat[ cols ]
    return feat

def feat_automl_user_cate_count(data):
    df = data.copy()
    feat = df[ ['index','user','road_item','item'] ]
    feat['user-road_item'] = feat['user'].astype('str') + '-' + feat['road_item'].astype('str')
    feat['user-item'] = feat['user'].astype('str') + '-' + feat['item'].astype('str')
    feat['user-road_item-item'] = feat['user'].astype('str') + '-' + feat['road_item'].astype('str') + '-' + feat['item'].astype('str')
    cate_list = [ 'user-road_item','user-item','user-road_item-item' ]
    cols = []
    for cate in cate_list:
        feat[cate+'_count'] = feat[ cate ].map( feat[ cate ].value_counts() )
        cols.append( cate+'_count' )    
    feat = feat[ cols ]
    return feat

def feat_u2i_road_item_time_diff(data):
    df = data.copy()
    feat =  df[['user','road_item_loc','road_item_time']]
    feat = feat.groupby(['user','road_item_loc']).first().reset_index()
    feat_group = feat.sort_values(['user','road_item_loc']).set_index(['user','road_item_loc']).groupby('user')
    
    feat1 = feat_group['road_item_time'].diff(1)
    feat2 = feat_group['road_item_time'].diff(-1)
    
    feat1.name = 'u2i_road_item_time_diff_history'
    feat2.name = 'u2i_road_item_time_diff_future'
    
    feat = df.merge(pd.concat([feat1,feat2],axis=1),how='left',on=['user','road_item_loc'])
    
    cols = [ 'u2i_road_item_time_diff_history', 'u2i_road_item_time_diff_future' ]
    feat = feat[ cols ]
    return feat

def feat_road_item_text_dot(data):
    df = data.copy()
    df = df[ ['index','road_item','item'] ]
    feat = df[ ['index'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    for k,v in item_feat.items():
        item_text[k] = v[0]
    
    def func(ss):
        item1 = ss['road_item']
        item2 = ss['item']
        if ( item1 in item_text ) and ( item2 in item_text ):
            item1_text = item_text[item1]
            item2_text = item_text[item2]
            c = np.dot( item1_text, item2_text )
            return c
        else:
            return np.nan
    feat['road_item_text_dot'] = df[ ['road_item','item'] ].apply(func, axis=1)
    feat = feat[ ['road_item_text_dot'] ]
    return feat

def feat_road_item_text_norm2(data):
    df = data.copy()
    df = df[ ['index','road_item','item'] ]
    feat = df[ ['index'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    for k,v in item_feat.items():
        item_text[k] = v[0]
    
    def func1(ss):
        item1 = ss['road_item']
        item2 = ss['item']
        if ( item1 in item_text ) and ( item2 in item_text ):
            item1_text = item_text[item1]
            item2_text = item_text[item2]
            a = np.linalg.norm( item1_text )
            b = np.linalg.norm( item2_text )
            return a*b
        else:
            return np.nan
        
    def func2(ss):
        item1 = ss
        if ( item1 in item_text ):
            item1_text = item_text[item1]
            a = np.linalg.norm( item1_text )
            return a
        else:
            return np.nan
        
    feat['road_item_text_product_norm2'] = df[ ['road_item','item'] ].apply(func1, axis=1)
    feat['road_item_text_norm2'] = df['road_item'].apply(func2)
    feat['item_text_norm2'] = df['item'].apply(func2)
    
    feat = feat[ ['road_item_text_product_norm2','road_item_text_norm2','item_text_norm2'] ]
    return feat

def feat_automl_cate_count_all_1(data):
    df = data.copy()
    categories = [ 'user','item','road_item','road_item_loc',
                  'query_item_loc','recall_type']
    feat = df[ ['index']+categories ]
    
    feat['loc_diff'] = df['query_item_loc']-df['road_item_loc']
    
    categories += ['loc_diff']
    
    n = len(categories)
    cols = []
    for a in range(n):
        cate1 = categories[a]
        feat[cate1+'_count_'] = feat[cate1].map( feat[cate1].value_counts() )
        cols.append( cate1+'_count_' )
        print(f'feat {cate1} fuck done')
    feat = feat[ cols ]
    return feat

def feat_automl_cate_count_all_2(data):
    df = data.copy()
    categories = [ 'user','item','road_item','road_item_loc',
                  'query_item_loc','recall_type']
    feat = df[ ['index']+categories ]
    
    feat['loc_diff'] = df['query_item_loc']-df['road_item_loc']
    
    categories += ['loc_diff']
    
    n = len(categories)
    cols = []
    for a in range(n):
        cate1 = categories[a]
        for b in range(a+1,n):
            cate2 = categories[b]
            name2 = f'{cate1}_{cate2}'
            
            feat_tmp = feat.groupby([cate1,cate2]).size()
            feat_tmp.name = f'{name2}_count_'
            feat = feat.merge(feat_tmp,how='left',on=[cate1,cate2])
            cols.append( name2+'_count_' )   
            print(f'feat {feat_tmp.name} fuck done')
    feat = feat[ cols ]
    return feat

def feat_automl_cate_count_all_3(data):
    df = data.copy()
    categories = [ 'user','item','road_item','road_item_loc',
                  'query_item_loc','recall_type']
    feat = df[ ['index']+categories ]
    
    feat['loc_diff'] = df['query_item_loc']-df['road_item_loc']
    
    categories += ['loc_diff']
    
    n = len(categories)
    cols = []
    for a in range(n):
        cate1 = categories[a]
        for b in range(a+1,n):
            cate2 = categories[b]
            for c in range(b+1,n):
                cate3 = categories[c]
                name3 = f'{cate1}_{cate2}_{cate3}'
                
                feat_tmp = feat.groupby([cate1,cate2,cate3]).size()
                feat_tmp.name = f'{name3}_count_'
                feat = feat.merge(feat_tmp,how='left',on=[cate1,cate2,cate3])
                cols.append( name3+'_count_' )
                print(f'feat {feat_tmp.name} fuck done')
    feat = feat[ cols ]
    return feat

def feat_time_window_cate_count(data):
    if mode=='valid':
        all_train_data = utils.load_pickle(all_train_data_path.format(cur_stage))
    else:
        all_train_data = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    item_with_time = all_train_data[["item_id", "time"]].sort_values(["item_id", "time"])
    item2time = item_with_time.groupby("item_id")["time"].agg(list).to_dict()
    utils.dump_pickle(item2time, item2time_path.format(mode))

    item2times = utils.load_pickle(item2time_path.format(mode))

    df = data.copy()
    df["item_time"] = df.set_index(["item", "time"]).index
    feat = df[["item_time"]]
    del df
    
    def find_count_around_time(item_time, mode, delta):
        item, t = item_time
        if mode == "left":
            left = t - delta
            right = t
        elif mode == "right":
            left = t
            right = t + delta
        else:
            left = t - delta
            right = t + delta
        click_times = item2times[item]
        count = 0
        for ts in click_times:
            if ts < left:
                continue
            elif ts > right:
                break
            else:
                count += 1
        return count
    
    feat["item_cnt_around_time_0.01"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.01))
    feat["item_cnt_before_time_0.01"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.01))
    feat["item_cnt_after_time_0.01"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.01))
    
    feat["item_cnt_around_time_0.02"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.02))
    feat["item_cnt_before_time_0.02"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.02))
    feat["item_cnt_after_time_0.02"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.02))
    
    feat["item_cnt_around_time_0.05"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.05))
    feat["item_cnt_before_time_0.05"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.05))
    feat["item_cnt_after_time_0.05"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.05))

    return feat[[
        "item_cnt_around_time_0.01", "item_cnt_before_time_0.01", "item_cnt_after_time_0.01",
        "item_cnt_around_time_0.02", "item_cnt_before_time_0.02", "item_cnt_after_time_0.02",
        "item_cnt_around_time_0.05", "item_cnt_before_time_0.05", "item_cnt_after_time_0.05",
    ]]

def feat_time_window_cate_count(data):
    # 做这个特征之前，先做一次item2time.py
    try:
        item2times = utils.load_pickle(item2time_path.format(mode, cur_stage))
    except:
        raise Exception("做这个特征之前，先做一次item2time.py")

    df = data.copy()
    df["item_time"] = df.set_index(["item", "time"]).index
    feat = df[["item_time"]]
    del df
    
    def find_count_around_time(item_time, mode, delta):
        item, t = item_time
        if mode == "left":
            left = t - delta
            right = t
        elif mode == "right":
            left = t
            right = t + delta
        else:
            left = t - delta
            right = t + delta
        click_times = item2times[item]
        count = 0
        for ts in click_times:
            if ts < left:
                continue
            elif ts > right:
                break
            else:
                count += 1
        return count
    
    feat["item_cnt_around_time_0.01"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.01))
    feat["item_cnt_before_time_0.01"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.01))
    feat["item_cnt_after_time_0.01"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.01))
    
    feat["item_cnt_around_time_0.02"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.02))
    feat["item_cnt_before_time_0.02"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.02))
    feat["item_cnt_after_time_0.02"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.02))
    
    feat["item_cnt_around_time_0.05"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.05))
    feat["item_cnt_before_time_0.05"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.05))
    feat["item_cnt_after_time_0.05"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.05))

    feat["item_cnt_around_time_0.07"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.07))
    feat["item_cnt_before_time_0.07"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.07))
    feat["item_cnt_after_time_0.07"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.07))

    feat["item_cnt_around_time_0.1"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.1))
    feat["item_cnt_before_time_0.1"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.1))
    feat["item_cnt_after_time_0.1"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.1))

    feat["item_cnt_around_time_0.15"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="all", delta=0.15))
    feat["item_cnt_before_time_0.15"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="left", delta=0.15))
    feat["item_cnt_after_time_0.15"] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode="right", delta=0.15))

    return feat[[
        "item_cnt_around_time_0.01", "item_cnt_before_time_0.01", "item_cnt_after_time_0.01",
        "item_cnt_around_time_0.02", "item_cnt_before_time_0.02", "item_cnt_after_time_0.02",
        "item_cnt_around_time_0.05", "item_cnt_before_time_0.05", "item_cnt_after_time_0.05",
        "item_cnt_around_time_0.07", "item_cnt_before_time_0.07", "item_cnt_after_time_0.07",
        "item_cnt_around_time_0.1", "item_cnt_before_time_0.1", "item_cnt_after_time_0.1",
        "item_cnt_around_time_0.15", "item_cnt_before_time_0.15", "item_cnt_after_time_0.15",
    ]]

#在召回集内，限定时间(qtime 附近) 这个item被召回了多少次
# item2times 改变了 其他的逻辑不变
def item_recall_cnt_around_qtime(data):
    item2times = data.groupby("item")["time"].agg(list).to_dict()

    df = data.copy()
    df["item_time"] = df.set_index(["item", "time"]).index
    feat = df[["item_time"]]
    del df

    def find_count_around_time(item_time, mode, delta):
        item, t = item_time
        if mode == "left":
            left = t - delta
            right = t
        elif mode == "right":
            left = t
            right = t + delta
        else:
            left = t - delta
            right = t + delta
        click_times = item2times[item]
        count = 0
        for ts in click_times:
            if ts < left:
                continue
            elif ts > right:
                break
            else:
                count += 1
        return count

    new_cols = []
    new_col_name = "item_recall_cnt_{}_time_{}"
    for delta in [0.01, 0.02, 0.05, 0.07, 0.1, 0.15]:
        print('running delta: ', delta)
        for mode in ["all", "left", "right"]:
            new_col = new_col_name.format(mode, delta)
            new_cols.append(new_col)
            feat[new_col] = feat["item_time"].apply(lambda x: find_count_around_time(x, mode=mode, delta=delta))

    return feat[new_cols]

def feat_automl_recall_type_cate_count(data):
    df = data.copy()
    
    feat = df[ ['index','item','road_item','recall_type'] ]
    feat['road_item-item'] = feat['road_item'].astype('str')+ '-' + feat['item'].astype('str')
    
    cols = []
    for cate1 in ['recall_type']:
        for cate2 in ['item','road_item','road_item-item']:
            name2 = f'{cate1}-{cate2}'
            
            feat_tmp = feat.groupby([cate1,cate2]).size()
            feat_tmp.name = f'{name2}_count'
            feat = feat.merge(feat_tmp,how='left',on=[cate1,cate2])
            cols.append( name2+'_count' )   
            print(f'feat {cate1} {cate2} fuck done')
    feat = feat[ cols ]
    return feat

def feat_automl_loc_diff_cate_count(data):
    df = data.copy()
    
    feat = df[ ['index','item','road_item','recall_type'] ]
    feat['road_item-item'] = feat['road_item'].astype('str')+ '-' + feat['item'].astype('str')
    
    
    feat['loc_diff'] = df['query_item_loc']-df['road_item_loc']
    
    cols = []
    for cate1 in ['loc_diff']:
        for cate2 in ['item','road_item','recall_type','road_item-item']:
            name2 = f'{cate1}-{cate2}'
            
            feat_tmp = feat.groupby([cate1,cate2]).size()
            feat_tmp.name = f'{name2}_count'
            feat = feat.merge(feat_tmp,how='left',on=[cate1,cate2])
            cols.append( name2+'_count' )   
            print(f'feat {cate1} {cate2} fuck done')
    feat = feat[ cols ]
    return feat

def feat_automl_user_and_recall_type_cate_count(data):
    df = data.copy()
    
    feat = df[ ['index','item','road_item','recall_type','user'] ]
    
    feat['road_item-item'] = feat['road_item'].astype('str') + '-' + feat['item'].astype('str')
    
    cols = []
    for cate1 in ['user']:
        for cate2 in ['recall_type']:
            for cate3 in ['item','road_item','road_item-item']:
                name3 = f'{cate1}-{cate2}-{cate3}'
                
                feat_tmp = feat.groupby([cate1,cate2,cate3]).size()
                feat_tmp.name = f'{name3}_count'
                feat = feat.merge(feat_tmp,how='left',on=[cate1,cate2,cate3])
                cols.append( name3+'_count' )   
                print(f'feat {cate1} {cate2} {cate3} fuck done')
    feat = feat[ cols ]
    return feat

def feat_i2i_cijs_topk_by_loc(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    topk = 3
    loc_bases = [0.9]
    for loc_base in loc_bases:
        print(f'Starting {loc_base}')
        result = {}
        result_topk_by_loc = {}
        result_history_loc_diff1_cnt = {}
        result_future_loc_diff1_cnt = {}
        result_history_loc_diff1_time_mean = {}
        result_future_loc_diff1_time_mean = {}
        
        for key in new_keys:
            if key not in i2i_sim_seq.keys():
                result[key] = np.nan
                continue
            result[key] = []
            result_history_loc_diff1_cnt[key] = 0.0
            result_future_loc_diff1_cnt[key] = 0.0
            
            result_history_loc_diff1_time_mean[key] = 0
            result_future_loc_diff1_time_mean[key] = 0
            
            records = i2i_sim_seq[key]
            if len(records)==0:
                print(key)
            for record in records:
                loc1, loc2, t1, t2, record_len = record
                if loc1-loc2>0:
                    if loc1-loc2==1:
                        result_history_loc_diff1_cnt[key] += 1
                        result_history_loc_diff1_time_mean[key] += (t1 - t2)
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (loc_base**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2        
                else:
                    if loc2-loc1==1:
                        result_future_loc_diff1_cnt[key] += 1
                        result_future_loc_diff1_time_mean[key] += (t2 - t1)
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (loc_base**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                result[key].append( (loc_diff,1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len))) 
                result_history_loc_diff1_time_mean[key] /=(result_history_loc_diff1_cnt[key]+1e-5)
                result_future_loc_diff1_time_mean[key] /=(result_future_loc_diff1_cnt[key]+1e-5)
                
            result_one = sorted(result[key],key=lambda x:x[0])
            result_one_len = len(result_one)
            
            result_topk_by_loc[key] = [x[1] for x in result_one[:topk]]+[np.nan]*max(0,topk-result_one_len)
            
        feat['history_loc_diff1_com_item_time_mean'] =  feat['new_keys'].map(result_history_loc_diff1_time_mean).fillna(0)
        feat['future_loc_diff1_com_item_time_mean'] =  feat['new_keys'].map(result_future_loc_diff1_time_mean).fillna(0)
        feat['history_loc_diff1_com_item_cnt'] =  feat['new_keys'].map(result_history_loc_diff1_cnt).fillna(0)
        feat['future_loc_diff1_com_item_cnt'] =  feat['new_keys'].map(result_future_loc_diff1_cnt).fillna(0)
        
        feat_top = []
        for key,value in result_topk_by_loc.items():
            feat_top.append([key[0],key[1]]+value)
        
        feat_top = pd.DataFrame(feat_top,columns=['road_item','item']+[f'i2i_cijs_top{k}_by_loc' for k in range(1,topk+1)])
        feat = feat.merge(feat_top,how='left',on=['road_item','item'])
    print('Finished getting result')
    cols = ['history_loc_diff1_com_item_time_mean',
            'future_loc_diff1_com_item_time_mean',
            'history_loc_diff1_com_item_cnt',
            'future_loc_diff1_com_item_cnt']+[f'i2i_cijs_top{k}_by_loc' for k in range(1,topk+1)]
    
    feat = feat[ cols ]
    return feat

def feat_i2i_cijs_median_mean_topk(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    topk = 3
    loc_bases = [0.9]
    for loc_base in loc_bases:
        print(f'Starting {loc_base}')
        result = {}
        result_median = {}
        result_mean = {}
        result_topk = {}
        for key in new_keys:
            if key not in i2i_sim_seq.keys():
                result[key] = np.nan
                continue
            result[key] = []
            records = i2i_sim_seq[key]
            if len(records)==0:
                print(key)
            for record in records:
                loc1, loc2, t1, t2, record_len = record
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (loc_base**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2        
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (loc_base**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                result[key].append( 1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len)) 
            result_one = sorted(result[key],reverse=True)
            result_one_len = len(result_one)
            
            result_median[key] = result_one[result_one_len//2] if result_one_len%2==1 else (result_one[result_one_len//2]+result_one[result_one_len//2-1])/2
            result_mean[key] = sum(result[key])/len(result[key])
            
            result_topk[key] = result_one[:topk]+[np.nan]*max(0,topk-result_one_len)
            
        feat['i2i_cijs_median'] = feat['new_keys'].map(result_median)
        feat['i2i_cijs_mean'] = feat['new_keys'].map(result_mean)
        feat_top = []
        for key,value in result_topk.items():
            feat_top.append([key[0],key[1]]+value)
        
        feat_top = pd.DataFrame(feat_top,columns=['road_item','item']+[f'i2i_cijs_top{k}_by_cij' for k in range(1,topk+1)])
        feat = feat.merge(feat_top,how='left',on=['road_item','item'])
    print('Finished getting result')
    cols = ['i2i_cijs_median','i2i_cijs_mean']+[f'i2i_cijs_top{k}_by_cij' for k in range(1,topk+1)]
    
    feat = feat[ cols ]
    return feat

def feat_different_type_road_score_sum_mean_by_item(data):  
    df = data.copy()
    feat = df[ ['user','item','index','sim_weight','recall_type'] ]
    
    cols = ['i2i_score','blend_score','i2i2i_score']#,'i2iw10_score','i2i2b_score']
    for i in range(len(cols)):
        feat[cols[i]] = feat['sim_weight']
        feat.loc[ feat['recall_type']!=i,cols[i] ] = np.nan
    
    for col in cols:
        df = feat[ ['item',col,'index'] ]
        df = df.groupby('item')[col].sum().reset_index()
        df[col+'_by_item_sum'] = df[col]
        df = df[ ['item',col+'_by_item_sum'] ]
        feat = pd.merge( feat, df, on='item', how='left')
        
        df = feat[ ['item',col,'index'] ]
        df = df.groupby('item')[col].mean().reset_index()
        df[col+'_by_item_mean'] = df[col]
        df = df[ ['item',col+'_by_item_mean'] ]
        feat = pd.merge( feat, df, on='item', how='left')
        
        
    feat = feat[[f'{i}_by_item_{j}' for i in cols for j in ['sum','mean']]]
    
    return feat

def feat_different_type_road_score_mean_by_road_item(data):  
    df = data.copy()
    feat = df[ ['user','road_item','index','sim_weight','recall_type'] ]
    
    cols = ['i2i_score','blend_score','i2i2i_score']#'i2iw10_score','i2i2b_score']
    for i in range(len(cols)):
        feat[cols[i]] = feat['sim_weight']
        feat.loc[ feat['recall_type']!=i,cols[i] ] = np.nan
    
    for col in cols:
        df = feat[ ['road_item',col,'index'] ]
        df = df.groupby('road_item')[col].mean().reset_index()
        df[col+'_by_road_item_mean'] = df[col]
        df = df[ ['road_item',col+'_by_road_item_mean'] ]
        feat = pd.merge( feat, df, on='road_item', how='left')
        
        
    feat = feat[[f'{i}_by_road_item_mean' for i in cols]]
    
    return feat


def feat_different_type_road_score_mean_by_loc_diff(data):  
    df = data.copy()
    feat = df[ ['user','index','sim_weight','recall_type'] ]
    feat['loc_diff'] = df['query_item_loc']-df['road_item_loc']
    
    cols = ['i2i_score','blend_score','i2i2i_score','i2iw10_score','i2i2b_score']
    for i in range(len(cols)):
        feat[cols[i]] = feat['sim_weight']
        feat.loc[ feat['recall_type']!=i,cols[i] ] = np.nan
    
    for col in cols:
        df = feat[ ['loc_diff',col,'index'] ]
        df = df.groupby('loc_diff')[col].mean().reset_index()
        df[col+'_by_loc_diff_mean'] = df[col]
        df = df[ ['loc_diff',col+'_by_loc_diff_mean'] ]
        feat = pd.merge( feat, df, on='loc_diff', how='left')
        
        
    feat = feat[[f'{i}_by_loc_diff_mean' for i in cols]]
    
    return feat

def feat_different_type_road_score_sum_mean_by_recall_type_and_item(data):  
    df = data.copy()
    feat = df[ ['user','item','index','sim_weight','recall_type'] ]
    
    cols = ['i2i_score','blend_score','i2i2i_score','i2iw10_score','i2i2b_score']
    for i in range(len(cols)):
        feat[cols[i]] = feat['sim_weight']
        feat.loc[ feat['recall_type']!=i,cols[i] ] = np.nan
    
    for col in cols:
        df = feat[ ['item','recall_type',col,'index'] ]
        df = df.groupby(['item','recall_type'])[col].sum().reset_index()
        df[col+'_by_item-recall_type_sum'] = df[col]
        df = df[ ['item','recall_type',col+'_by_item-recall_type_sum'] ]
        feat = pd.merge( feat, df, on=['item','recall_type'], how='left')
        
        df = feat[ ['item','recall_type',col,'index'] ]
        df = df.groupby(['item','recall_type'])[col].mean().reset_index()
        df[col+'_by_item-recall_type_mean'] = df[col]
        df = df[ ['item','recall_type',col+'_by_item-recall_type_mean'] ]
        feat = pd.merge( feat, df, on=['item','recall_type'], how='left')
        
        
    feat = feat[[f'{i}_by_item-recall_type_{j}' for i in cols for j in ['sum','mean']]]
    
    return feat


def feat_base_info_in_stage(data):
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    #all_train_stage_data = pd.concat( all_train_stage_data.iloc[0:1000], all_train_stage_data.iloc[-10000:] )
    df_train_stage = all_train_stage_data

    df = data.copy()
    feat = df[ ['index','road_item','item','stage'] ]
    stage2sim_item = {}
    stage2item_cnt = {}
    stage2com_item_cnt = {}
    for sta in range(cur_stage+1):
        df_train = df_train_stage[ df_train_stage['stage']==sta ]
        user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
        user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
        user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
        user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
        
        sim_item = {}
        item_cnt = defaultdict(int)
        com_item_cnt = {}
        for user, items in user_item_dict.items():

            times = user_time_dict[user]
            for loc1, item in enumerate(items):
                item_cnt[item] += 1
                sim_item.setdefault(item, {})
                com_item_cnt.setdefault(item, {})
                for loc2, relate_item in enumerate(items):  
                    if item == relate_item:
                        continue
                    
                    t1 = times[loc1]
                    t2 = times[loc2]
                    sim_item[item].setdefault(relate_item, 0)
                    com_item_cnt[item].setdefault(relate_item, 0)
                    
                    if loc1-loc2>0:
                        time_weight = (1 - (t1 - t2) * 100)
                        if time_weight<=0.2:
                            time_weight = 0.2
                            
                        loc_diff = loc1-loc2-1
                        loc_weight = (0.9**loc_diff)
                        if loc_weight <= 0.2:
                            loc_weight = 0.2

                        sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items))
                            
                    else:
                        time_weight = (1 - (t2 - t1) * 100)
                        if time_weight<=0.2:
                            time_weight = 0.2
                        
                        loc_diff =  loc2-loc1-1
                        loc_weight =  (0.9**loc_diff)
                        
                        if loc_weight <= 0.2:
                            loc_weight = 0.2
                        
                        sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items)) 
                    com_item_cnt[item][relate_item] += 1.0

        stage2sim_item[sta] = sim_item
        stage2item_cnt[sta] = item_cnt
        stage2com_item_cnt[sta] = com_item_cnt


    sta_list = []
    itemb_list = []
    sum_sim_list = []
    count_sim_list = []
    mean_sim_list = []
    nunique_itema_count_list = []


    for sta in range(cur_stage+1):
        for key1 in stage2sim_item[sta].keys():
            val = 0
            count = 0
            for key2 in stage2sim_item[sta][key1].keys():
                val += stage2sim_item[sta][key1][key2]
                count += stage2com_item_cnt[sta][key1][key2]

            sta_list.append( sta )
            itemb_list.append( key1 )
            sum_sim_list.append( val )
            count_sim_list.append( count )
            mean_sim_list.append( val/count )
            nunique_itema_count_list.append( len( stage2sim_item[sta][key1].keys() ) )

    data1 = pd.DataFrame( {'stage':sta_list, 'item':itemb_list, 'sum_sim_in_stage':sum_sim_list, 'count_sim_in_stage':count_sim_list, 
                           'mean_sim_in_stage':mean_sim_list, 'nunique_itema_count_in_stage':nunique_itema_count_list } )
    '''
    sta_list = []
    item_list = []
    cnt_list = []
    for sta in range(cur_stage+1):
        for key1 in stage2item_cnt[sta].keys():
            sta_list.append(sta)
            item_list.append(key1)
            cnt_list.append( stage2item_cnt[sta][key1] )

    data2 = pd.DataFrame( {'stage':sta_list, 'road_item':item_list, 'stage_road_item_cnt':cnt_list } )
    data3 = pd.DataFrame( {'stage':sta_list, 'item':item_list, 'stage_item_cnt':cnt_list } )
    '''
    #feat = pd.merge( feat,data1, how='left',on=['stage','road_item','item'] )
    #feat = pd.merge( feat,data2, how='left',on=['stage','road_item'] )
    feat = pd.merge( feat,data1, how='left',on=['stage','item'] )
    feat = feat[ ['sum_sim_in_stage','count_sim_in_stage','mean_sim_in_stage','nunique_itema_count_in_stage'] ]
    return feat

def feat_item_time_info_in_stage(data):
    df = data.copy()
    feat = df[ ['index','item','stage','time'] ]
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    df_train_stage = all_train_stage_data
    data1 = df_train_stage.groupby( ['stage','item_id'] )['time'].agg( ['max','min','mean'] ).reset_index()
    data1.columns = [ 'stage','item','time_max_in_stage','time_min_in_stage','time_mean_in_stage' ]
    data1['time_dura_in_stage'] = data1['time_max_in_stage'] - data1['time_min_in_stage']
    feat = pd.merge( feat,data1, how='left',on=['stage','item'] )
    feat['time_diff_min_in_stage'] = feat['time'] - feat['time_min_in_stage']
    feat['time_diff_max_in_stage'] = feat['time_max_in_stage'] - feat['time'] 
    cols = [ 'time_dura_in_stage','time_max_in_stage','time_min_in_stage','time_mean_in_stage','time_diff_min_in_stage','time_diff_max_in_stage' ]
    feat = feat[ cols ]
    return feat

def feat_user_info_in_stage(data):
    df = data.copy()
    feat = df[ ['index','item','user','stage'] ]
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    df_train_stage = all_train_stage_data
    data1 = df_train_stage.groupby( ['stage','user_id'] )['index'].count()
    data1.name = 'user_count_in_stage'
    data1 = data1.reset_index()
    data1 = data1.rename( columns={'user_id':'user'} )
    data2 = df_train_stage.groupby( ['stage','item_id'] )['user_id'].nunique()
    data2.name = 'item_nunique_in_stage'
    data2 = data2.reset_index()
    data2 = data2.rename( columns={'item_id':'item'} )
    data3 = df_train_stage.groupby( ['stage','item_id'] )['user_id'].count()
    data3.name = 'item_count_in_stage'
    data3 = data3.reset_index()
    data3 = data3.rename( columns={'item_id':'item'} )
    data3[ 'item_ratio_in_stage' ] = data3[ 'item_count_in_stage' ]  / data2['item_nunique_in_stage']
     
    feat = pd.merge( feat,data1, how='left',on=['stage','user'] )
    feat = pd.merge( feat,data2, how='left',on=['stage','item'] )
    feat = pd.merge( feat,data3, how='left',on=['stage','item'] )
    cols = [ 'user_count_in_stage','item_nunique_in_stage','item_ratio_in_stage' ]
    feat = feat[ cols ]
    return feat
    
def feat_item_com_cnt_in_stage(data):
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    item_stage_cnt = all_train_stage_data.groupby(["item_id"])["stage"].value_counts().to_dict()
    feat = data[["road_item", "stage"]]
    feat["head"] = feat.set_index(["road_item", "stage"]).index
    feat["itema_cnt_in_stage"] = feat["head"].map(item_stage_cnt)
    return feat[["itema_cnt_in_stage"]]

def item_cnt_in_stage2(data):
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    item_stage_cnt = all_train_stage_data.groupby(["item_id"])["stage"].value_counts().to_dict()
    feat = data[["item", "stage"]]
    feat["head"] = feat.set_index(["item", "stage"]).index
    feat["item_stage_cnt"] = feat["head"].map(item_stage_cnt)
    return feat[["item_stage_cnt"]]

def feat_item_cnt_in_different_stage(data):
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    feat = data[["item"]]
    cols = []
    for sta in range(cur_stage+1):
        train_stage_data = all_train_stage_data[ all_train_stage_data['stage']==sta ]
        item_stage_cnt = train_stage_data.groupby(['item_id'])['index'].count()
        item_stage_cnt.name = f"item_stage_cnt_{sta}"
        item_stage_cnt = item_stage_cnt.reset_index()
        item_stage_cnt.columns = ['item',f"item_stage_cnt_{sta}"]
        feat = pd.merge( feat,item_stage_cnt,how='left',on='item' )
        cols.append( f"item_stage_cnt_{sta}" )
    #import pdb
    #pdb.set_trace()
    return feat[ cols ] 

def feat_user_cnt_in_different_stage(data):
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    feat = data[["user"]]
    cols = []
    for sta in range(cur_stage+1):
        train_stage_data = all_train_stage_data[ all_train_stage_data['stage']==sta ]
        user_stage_cnt = train_stage_data.groupby(['user_id'])['index'].count()
        user_stage_cnt.name = f"user_stage_cnt_{sta}"
        user_stage_cnt = user_stage_cnt.reset_index()
        user_stage_cnt.columns = ['user',f"user_stage_cnt_{sta}"]
        feat = pd.merge( feat,user_stage_cnt,how='left',on='user' )
        cols.append( f"user_stage_cnt_{sta}" )
    #import pdb
    #pdb.set_trace()
    return feat[ cols ]  

def feat_user_and_item_count_in_three_init_data(data):
    df = data.copy()
    feat = df[ ['index','item','user','stage'] ]
    if mode=='valid':
        df_train_stage = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    else:
        df_train_stage = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    data1 = df_train_stage.groupby( ['stage','item_id'] )['index'].count()
    data1.name = 'in_stage_item_count'
    data1 = data1.reset_index()
    data1 = data1.rename( columns = {'item_id':'item'} )

    data2 = df_train_stage.groupby( ['stage','user_id'] )['index'].count()
    data2.name = 'in_stage_user_count'
    data2 = data2.reset_index()
    data2 = data2.rename( columns = {'user_id':'user'} )

    data3 = df_train_stage.groupby( ['item_id'] )['index'].count()
    data3.name = 'no_in_stage_item_count'
    data3 = data3.reset_index()
    data3 = data3.rename( columns = {'item_id':'item'} )

    data4 = df_train_stage.groupby( ['user_id'] )['index'].count()
    data4.name = 'no_in_stage_user_count'
    data4 = data4.reset_index()
    data4 = data4.rename( columns = {'user_id':'user'} )

    data5 = df_train.groupby( ['item_id'] )['index'].count()
    data5.name = 'no_stage_item_count'
    data5 = data5.reset_index()
    data5 = data5.rename( columns = {'item_id':'item'} )

    data6 = df_train.groupby( ['user_id'] )['index'].count()
    data6.name = 'no_stage_user_count'
    data6 = data6.reset_index()
    data6 = data6.rename( columns = {'user_id':'user'} )

    feat = pd.merge( feat,data1,how='left',on=['stage','item'] )
    feat = pd.merge( feat,data2,how='left',on=['stage','user'] )
    feat = pd.merge( feat,data3,how='left',on=['item'] )
    feat = pd.merge( feat,data4,how='left',on=['user'] )
    feat = pd.merge( feat,data5,how='left',on=['item'] )
    feat = pd.merge( feat,data6,how='left',on=['user'] )

    cols = [ 'in_stage_item_count','in_stage_user_count','no_in_stage_item_count','no_in_stage_user_count','no_stage_item_count','no_stage_user_count' ]
    return feat[ cols ]
#def feat_item_count_in_three_init_data(data):

def feat_i2i2i_sim(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i2i_sim_seq')
    i2i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')
    print('Getting result')
    
    result = np.zeros((len(new_keys),4))
    
    item_cnt = df_train['item_id'].value_counts().to_dict()
    
    for i in range(len(new_keys)):
        key = new_keys[i]
        if key not in i2i2i_sim_seq.keys():
            continue  
        records = i2i2i_sim_seq[key]
        result[i,0] = len(records)
        
        if len(records)==0:
            print(key)
        for record in records:
            item,score1_1,score1_2,score2_1,score2_2 = record
            result[i,1] += score1_1*score1_2
            result[i,2] += score2_1*score2_2
            result[i,3] += item_cnt[item]
        
    result[:,1]/=(result[i,0]+1e-9)
    result[:,2]/=(result[i,0]+1e-9)
    result[:,3]/=(result[i,0]+1e-9)
        
    print('Finished getting result')
    cols = ['i2i2i_road_cnt','i2i2i_score1_mean','i2i2i_score2_mean','i2i2i_middle_item_cnt_mean']
    result = pd.DataFrame(result,index=new_keys,columns=cols)
    result = result.reset_index()
    result.rename(columns={'index':'new_keys'},inplace=True)
    
    feat = feat.merge(result,how='left',on='new_keys')
    
    feat = feat[ cols ]
    return feat

def feat_i2i2b_sim(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i2b_sim_seq')
    i2i2b_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i2b_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i2b_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')
    print('Getting result')
    
    result = np.zeros((len(new_keys),4))
    
    item_cnt = df_train['item_id'].value_counts().to_dict()
    
    for i in range(len(new_keys)):
        key = new_keys[i]
        if key not in i2i2b_sim_seq.keys():
            continue
        
        records = i2i2b_sim_seq[key]
        result[i,0] = len(records)
        
        if len(records)==0:
            print(key)
        for record in records:
            item,score1_1,score1_2,score2_1,score2_2 = record
            result[i,1] += score1_1*score1_2
            result[i,2] += score2_1*score2_2
            result[i,3] += item_cnt[item]
        
    result[:,1]/=(result[i,0]+1e-9)
    result[:,2]/=(result[i,0]+1e-9)
    result[:,3]/=(result[i,0]+1e-9)
        
    print('Finished getting result')
    cols = ['i2i2b_road_cnt','i2i2b_score1_mean','i2i2b_score2_mean','i2i2b_middle_item_cnt_mean']
    result = pd.DataFrame(result,index=new_keys,columns=cols)
    result = result.reset_index()
    result.rename(columns={'index':'new_keys'},inplace=True)
    
    feat = feat.merge(result,how='left',on='new_keys')
    
    feat = feat[ cols ]
    return feat

def feat_numerical_groupby_item_cnt_in_stage(data):
    df = data.copy()
    num_cols = [ 'sim_weight', 'loc_weight', 'time_weight', 'rank_weight' ]
    cate_col = 'item_stage_cnt'
    feat = df[ ['index','road_item','item'] ]
    feat1 = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'item_cnt_in_stage2_{mode}_{cur_stage}.pkl') )
    df[ cate_col ] = feat1[ cate_col ]
    feat[ cate_col ] = feat1[ cate_col ]
    cols = []
    for col in num_cols:
        t = df.groupby(cate_col)[col].agg( ['mean','max','min'] )
        cols += [ f'{col}_{i}_groupby_{cate_col}' for i in ['mean','max','min'] ]
        t.columns = [ f'{col}_{i}_groupby_{cate_col}' for i in ['mean','max','min'] ]
        t = t.reset_index()
        feat = pd.merge( feat, t, how='left', on=cate_col )

    return feat[ cols ]
    
#i2i_score,
# 
def feat_item_stage_nunique(data):
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    item_stage_nunique = all_train_stage_data.groupby(["item_id"])["stage"].nunique()
    feat = data[["item"]]
    feat["item_stage_nunique"] = feat["item"].map(item_stage_nunique)
    return feat[["item_stage_nunique"]]

def feat_item_qtime_time_diff(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    
    item_time_list = df_train.sort_values('time').groupby('item_id',sort=False)['time'].agg(list)
    
    df = data.copy()
    
    feat = df[['item','query_item_time']]
    
    df_v = feat.values
    result_history = np.zeros(df_v.shape[0])*np.nan
    result_future = np.zeros(df_v.shape[0])*np.nan
    
    for i in range(df_v.shape[0]):
        time = df_v[i,1]
        time_list = [0]+item_time_list[df_v[i,0]]+[1]
        for j in range(1,len(time_list)):
            if time<time_list[j]:
                result_future[i] = time_list[j]-time
                result_history[i] = time-time_list[j-1]
                break
    
    feat['item_qtime_time_diff_history'] = result_history
    feat['item_qtime_time_diff_future'] = result_future
    
    return feat[['item_qtime_time_diff_history','item_qtime_time_diff_future']]


def feat_item_cumcount(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    
    item_time_list = df_train.sort_values('time').groupby('item_id',sort=False)['time'].agg(list)
    
    df = data.copy()
    
    feat = df[['item','query_item_time']]
    
    df_v = feat.values
    result = np.zeros(df_v.shape[0])
    
    for i in range(df_v.shape[0]):
        time = df_v[i,1]
        time_list = item_time_list[df_v[i,0]]+[1]
        for j in range(len(time_list)):
            if time<time_list[j]:
                result[i] = j
                break
    
    feat['item_cumcount'] = result
    
    feat['item_cumrate'] = feat['item_cumcount']/feat['item'].map(df_train['item_id'].value_counts()).fillna(1e-5)
    
    return feat[['item_cumcount','item_cumrate']]

def feat_road_time_bins_cate_cnt(data):
    df = data.copy()
    
    categoricals = ['item','road_item','user','recall_type']
    feat = df[['road_item_time']+categoricals]
    feat['loc_diff'] = df['query_item_loc']-df['road_item_loc']
    categoricals.append('loc_diff')
    
    feat['road_time_bins'] = pd.Categorical(pd.cut(feat['road_item_time'],100)).codes
    
    cols = []
    for cate in categoricals:
        cnt = feat.groupby([cate,'road_time_bins']).size()
        cnt.name = f'{cate}_cnt_by_road_time_bins'
        cols.append(cnt.name)
        feat = feat.merge(cnt,how='left',on=[cate,'road_time_bins'])
    
    return feat[cols]


def feat_time_window_cate_count(data):
    # 做这个特征之前，先做一次item2time.py
    import time as ti
    t = ti.time()
    df = data.copy()
    
    feat = df[['item','query_item_time']]
    df_v = feat.values
    
    del df
    
    try:
        item_time_list = utils.load_pickle(item2time_path.format(mode, cur_stage))
    except:
        raise Exception("做这个特征之前，先做一次item2time.py")
    
    
    delta_list = np.array(sorted([0.01, 0.02, 0.05, 0.07, 0.1, 0.15]))
    delta_list2 = delta_list[::-1]
    
    delta_n = delta_list.shape[0]
    n = delta_n*2+1
    
    result_tmp = np.zeros((df_v.shape[0],n))
    result_equal = np.zeros(df_v.shape[0])
    
    for i in range(df_v.shape[0]):
        time = np.ones(n)*df_v[i,1]
        time[:delta_n] -= delta_list2
        time[-delta_n:] += delta_list
        
        time_list = item_time_list[df_v[i,0]]+[10]
        
        k = 0
        for j in range(len(time_list)):
            while k<n and time[k]<time_list[j] :
                result_tmp[i,k] = j
                k += 1
                
            if time[delta_n]==time_list[j]:
                result_equal[i] += 1
        result_tmp[i,k:] = j
        
        if i%100000 == 0:
            print(f'[{i}/{df_v.shape[0]}]:time {ti.time()-t:.3f}s')
            t = ti.time()
    
    result = np.zeros((df_v.shape[0],delta_n*3))
    
    for i in range(delta_n):
        result[:,i*3+0] = result_tmp[:,delta_n] - result_tmp[:,i]
        result[:,i*3+1] = result_tmp[:,-(i+1)] - result_tmp[:,delta_n] + result_equal
        result[:,i*3+2] = result_tmp[:,-(i+1)] - result_tmp[:,i]
    
    
    cols = [f'item_cnt_{j}_time_{i}' for i in delta_list2 for j in ['before','after','around']]

    
    result = pd.DataFrame(result,columns=cols)
    
    result = result[[
        "item_cnt_around_time_0.01", "item_cnt_before_time_0.01", "item_cnt_after_time_0.01",
        "item_cnt_around_time_0.02", "item_cnt_before_time_0.02", "item_cnt_after_time_0.02",
        "item_cnt_around_time_0.05", "item_cnt_before_time_0.05", "item_cnt_after_time_0.05",
        "item_cnt_around_time_0.07", "item_cnt_before_time_0.07", "item_cnt_after_time_0.07",
        "item_cnt_around_time_0.1", "item_cnt_before_time_0.1", "item_cnt_after_time_0.1",
        "item_cnt_around_time_0.15", "item_cnt_before_time_0.15", "item_cnt_after_time_0.15",
    ]]
    
    return result


    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = max(item_feat.keys())+1
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[0]
    
    item_l2 = np.linalg.norm(item_np,axis=1)
    
    n = feat.shape[0]
    
    result = np.zeros((n,3))
    
    result[:,1] = item_l2[feat['road_item']]
    result[:,2] = item_l2[feat['item']]
    result[:,0] = result[:,1]*result[:,2]
    
    feat['road_item_text_product_norm2'] = result[:,0]
    feat['road_item_text_norm2'] = result[:,1]
    feat['item_text_norm2'] = result[:,2]
    
    feat.loc[(~feat['item'].isin(item_feat.keys()))|(~feat['road_item'].isin(item_feat.keys())),'road_item_text_product_norm2'] = np.nan
    feat.loc[(~feat['road_item'].isin(item_feat.keys())),'road_item_text_norm2'] = np.nan
    feat.loc[(~feat['item'].isin(item_feat.keys())),'item_text_norm2'] = np.nan
    
    feat = feat[ ['road_item_text_product_norm2','road_item_text_norm2','item_text_norm2'] ]
    return feat


def feat_road_item_text_cossim(data):
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[0]
    
    item_l2 = np.linalg.norm(item_np,axis=1)
    
    n = feat.shape[0]
    
    result = np.zeros(n)
    
    batch_size = 100000
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    for i in range(batch_num):
        result[i*batch_size:(i+1)*batch_size] = np.multiply(item_np[feat['road_item'][i*batch_size:(i+1)*batch_size],:],item_np[feat['item'][i*batch_size:(i+1)*batch_size],:]).sum(axis=1)
    
    result = np.divide(result,item_l2[feat['road_item']]*item_l2[feat['item']]+1e-9)
    
    feat['road_item_text_cossim'] = result
    
    feat.loc[(~feat['item'].isin(item_feat.keys()))|(~feat['road_item'].isin(item_feat.keys())),'road_item_text_cossim'] = np.nan

    return feat[['road_item_text_cossim']]

def feat_road_item_text_eulasim(data):
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[0]
    
    n = feat.shape[0]
    
    result = np.zeros(n)
    
    batch_size = 100000
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    for i in range(batch_num):
        result[i*batch_size:(i+1)*batch_size] = np.linalg.norm(item_np[feat['road_item'][i*batch_size:(i+1)*batch_size],:]-item_np[feat['item'][i*batch_size:(i+1)*batch_size],:],axis=1)
    
    feat['road_item_text_eulasim'] = result
    
    feat.loc[(~feat['item'].isin(item_feat.keys()))|(~feat['road_item'].isin(item_feat.keys())),'road_item_text_eulasim'] = np.nan

    return feat[['road_item_text_eulasim']]

def feat_road_item_text_dot(data):
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[0]
    
    item_l2 = np.linalg.norm(item_np,axis=1)
    
    n = feat.shape[0]
    
    result = np.zeros(n)
    
    batch_size = 100000
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    for i in range(batch_num):
        result[i*batch_size:(i+1)*batch_size] = np.multiply(item_np[feat['road_item'][i*batch_size:(i+1)*batch_size],:],item_np[feat['item'][i*batch_size:(i+1)*batch_size],:]).sum(axis=1)
    
    feat['road_item_text_dot'] = result
    feat.loc[(~feat['item'].isin(item_feat.keys()))|(~feat['road_item'].isin(item_feat.keys())),'road_item_text_dot'] = np.nan

    return feat[['road_item_text_dot']]

def feat_road_item_text_norm2(data):
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[0]
    
    item_l2 = np.linalg.norm(item_np,axis=1)
    
    n = feat.shape[0]
    
    result = np.zeros((n,3))
    
    result[:,1] = item_l2[feat['road_item']]
    result[:,2] = item_l2[feat['item']]
    result[:,0] = result[:,1]*result[:,2]
    
    feat['road_item_text_product_norm2'] = result[:,0]
    feat['road_item_text_norm2'] = result[:,1]
    feat['item_text_norm2'] = result[:,2]
    
    feat.loc[(~feat['item'].isin(item_feat.keys()))|(~feat['road_item'].isin(item_feat.keys())),'road_item_text_product_norm2'] = np.nan
    feat.loc[(~feat['road_item'].isin(item_feat.keys())),'road_item_text_norm2'] = np.nan
    feat.loc[(~feat['item'].isin(item_feat.keys())),'item_text_norm2'] = np.nan
    
    feat = feat[ ['road_item_text_product_norm2','road_item_text_norm2','item_text_norm2'] ]
    return feat

def feat_i2i_cijs_topk_by_loc(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    topk = 3
    loc_base = 0.9
    
    print(f'Starting {loc_base}')
    result = np.zeros((len(new_keys),4+topk))
    
    for i in range(len(new_keys)):
        key = new_keys[i]
        if key not in i2i_sim_seq.keys():
            result[i,:] = np.nan
            continue
        
        records = i2i_sim_seq[key]
        if len(records)==0:
            print(key)
            
        result_one = []
        
        for record in records:
            loc1, loc2, t1, t2, record_len = record
            if loc1-loc2>0:
                if loc1-loc2==1:
                    result[i,2] += 1
                    result[i,0] += (t1 - t2)
                    
                time_weight = (1 - (t1 - t2) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                    
                loc_diff = loc1-loc2-1
                loc_weight = (loc_base**loc_diff)
                if loc_weight <= 0.2:
                    loc_weight = 0.2        
            else:
                if loc2-loc1==1:
                    result[i,3] += 1
                    result[i,1] += (t2 - t1)
                    
                time_weight = (1 - (t2 - t1) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                
                loc_diff =  loc2-loc1-1
                loc_weight =  (loc_base**loc_diff)
                
                if loc_weight <= 0.2:
                    loc_weight = 0.2
                
            result_one.append( (loc_diff,1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len)) )
            result[i,1]/=(result[i,3]+1e-5)
            result[i,0]/=(result[i,2]+1e-5)
            
        result_one = sorted(result_one,key=lambda x:x[0])
        result_one_len = len(result_one)
        
        result[i,4:] = [x[1] for x in result_one[:topk]]+[np.nan]*max(0,topk-result_one_len)
    
    cols = ['history_loc_diff1_com_item_time_mean',
            'future_loc_diff1_com_item_time_mean',
            'history_loc_diff1_com_item_cnt',
            'future_loc_diff1_com_item_cnt']+[f'i2i_cijs_top{k}_by_loc' for k in range(1,topk+1)]
    
    result = pd.DataFrame(result,columns=cols,index=new_keys)
    result = result.reset_index()
    result.rename(columns={'index':'new_keys'},inplace=True)
    
    feat = feat.merge(result,how='left',on='new_keys')
    
    print('Finished getting result')
        
    feat = feat[ cols ]
    return feat

def feat_i2i_cijs_topk_by_loc(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    topk = 3
    loc_base = 0.9
    
    print(f'Starting {loc_base}')
    result = np.zeros((len(new_keys),4+topk))
    
    for i in range(len(new_keys)):
        key = new_keys[i]
        if key not in i2i_sim_seq.keys():
            result[i,:] = np.nan
            #result[i] = np.nan
            continue
        
        records = i2i_sim_seq[key]
        if len(records)==0:
            print(key)
            
        result_one = []
        
        for record in records:
            loc1, loc2, t1, t2, record_len = record
            if loc1-loc2>0:
                if loc1-loc2==1:
                    result[i,2] += 1
                    result[i,0] += (t1 - t2)
                    
                time_weight = (1 - (t1 - t2) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                    
                loc_diff = loc1-loc2-1
                loc_weight = (loc_base**loc_diff)
                if loc_weight <= 0.2:
                    loc_weight = 0.2        
            else:
                if loc2-loc1==1:
                    result[i,3] += 1
                    result[i,1] += (t2 - t1)
                    
                time_weight = (1 - (t2 - t1) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                
                loc_diff =  loc2-loc1-1
                loc_weight =  (loc_base**loc_diff)
                
                if loc_weight <= 0.2:
                    loc_weight = 0.2
                
            result_one.append( (loc_diff,1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len)) )
            result[i,1]/=(result[i,3]+1e-5)
            result[i,0]/=(result[i,2]+1e-5)
            
        result_one = sorted(result_one,key=lambda x:x[0])
        result_one_len = len(result_one)
        
        result[i,4:] = [x[1] for x in result_one[:topk]] + [np.nan]*max(0,topk-result_one_len)

    
    cols = ['history_loc_diff1_com_item_time_mean',
            'future_loc_diff1_com_item_time_mean',
            'history_loc_diff1_com_item_cnt',
            'future_loc_diff1_com_item_cnt']+[f'i2i_cijs_top{k}_by_loc' for k in range(1,topk+1)]
    
    result = pd.DataFrame(result,columns=cols,index=new_keys)
    result = result.reset_index()
    result.rename(columns={'index':'new_keys'},inplace=True)
    
    feat = feat.merge(result,how='left',on='new_keys')
    
    print('Finished getting result')
        
    feat = feat[ cols ]
    return feat

def feat_i2i_cijs_median_mean_topk(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    topk = 3
    loc_base = 0.9
    print(f'Starting {loc_base}')

    #median,mean,topk
    result = np.zeros((len(new_keys),2+topk))
    
    for i in range(len(new_keys)):
        key = new_keys[i]
        if key not in i2i_sim_seq.keys():
            result[i,:] = np.nan
            continue
        
        records = i2i_sim_seq[key]
        if len(records)==0:
            print(key)
            
        result_one = []
        
        for record in records:
            loc1, loc2, t1, t2, record_len = record
            if loc1-loc2>0:
                time_weight = (1 - (t1 - t2) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                    
                loc_diff = loc1-loc2-1
                loc_weight = (loc_base**loc_diff)
                if loc_weight <= 0.2:
                    loc_weight = 0.2        
            else:
                time_weight = (1 - (t2 - t1) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                
                loc_diff =  loc2-loc1-1
                loc_weight =  (loc_base**loc_diff)
                
                if loc_weight <= 0.2:
                    loc_weight = 0.2
                
            result_one.append( 1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len)) 
        result_one = sorted(result_one,reverse=True)
        result_one_len = len(result_one)
        
        result[i,0] = result_one[result_one_len//2] if result_one_len%2==1 else (result_one[result_one_len//2]+result_one[result_one_len//2-1])/2
        result[i,1] = sum(result_one)/(len(result_one))
        
        result[i,2:] = result_one[:topk]+[np.nan]*max(0,topk-result_one_len)
    
    cols = ['i2i_cijs_median','i2i_cijs_mean']+[f'i2i_cijs_top{k}_by_cij' for k in range(1,topk+1)]
    
    result = pd.DataFrame(result,columns=cols,index=new_keys)
    result = result.reset_index()
    result.rename(columns={'index':'new_keys'},inplace=True)
    
    feat = feat.merge(result,how='left',on='new_keys')
    
    print('Finished getting result')
    
    feat = feat[ cols ]
    return feat


def feat_different_type_road_score_sum_mean(data):  
    df = data.copy()
    feat = df[ ['user','item','index','sim_weight','recall_type'] ]
    feat['i2i_score'] = feat['sim_weight']
    feat['blend_score'] = feat['sim_weight']
    feat['i2i2i_score'] = feat['sim_weight']
    feat.loc[ feat['recall_type']!=0 , 'i2i_score'] = np.nan
    feat.loc[ feat['recall_type']!=1 , 'blend_score'] = np.nan
    feat.loc[ feat['recall_type']!=2 , 'i2i2i_score'] = np.nan

    df = feat[ ['index','user','item','i2i_score','blend_score','i2i2i_score'] ]
    df = df.groupby( ['user','item'] )[ ['i2i_score','blend_score','i2i2i_score'] ].agg( ['sum','mean'] ).reset_index()
    df.columns = ['user','item'] + [ f'{i}_{j}' for i in ['i2i_score','blend_score','i2i2i_score'] for j in ['sum','mean'] ]
    feat = pd.merge( feat, df, on=['user','item'], how='left')
    feat = feat[ ['i2i_score','i2i_score_sum','i2i_score_mean',
                  'blend_score','blend_score_sum','blend_score_mean',
                  'i2i2i_score','i2i2i_score_sum','i2i2i_score_mean',] ]
    return feat

def feat_automl_recall_type_cate_count(data):
    df = data.copy()
    
    feat = df[ ['index','item','road_item','recall_type'] ]
    
    cols = []
    for cate1 in ['recall_type']:
        for cate2 in ['item','road_item']:
            name2 = f'{cate1}-{cate2}'
            
            feat_tmp = feat.groupby([cate1,cate2]).size()
            feat_tmp.name = f'{name2}_count'
            feat = feat.merge(feat_tmp,how='left',on=[cate1,cate2])
            cols.append( name2+'_count' )   
            print(f'feat {cate1} {cate2} fuck done')
            
    tmp = feat.groupby(['recall_type','road_item','item']).size()
    tmp.name = 'recall_type-road_item-item_count'
    feat = feat.merge(tmp,how='left',on=['recall_type','road_item','item'])
    cols.append(tmp.name)
    
    print('feat recall_type road_item item fuck done')
            
    feat = feat[ cols ]
    
    return feat

def feat_automl_loc_diff_cate_count(data):
    df = data.copy()
    
    feat = df[ ['index','item','road_item','recall_type'] ]
    
    feat['loc_diff'] = df['query_item_loc']-df['road_item_loc']
    
    cols = []
    for cate1 in ['loc_diff']:
        for cate2 in ['item','road_item','recall_type']:
            name2 = f'{cate1}-{cate2}'
            
            feat_tmp = feat.groupby([cate1,cate2]).size()
            feat_tmp.name = f'{name2}_count'
            feat = feat.merge(feat_tmp,how='left',on=[cate1,cate2])
            cols.append( name2+'_count' )   
            print(f'feat {cate1} {cate2} fuck done')
            
    tmp = feat.groupby(['loc_diff','road_item','item']).size()
    tmp.name = 'loc_diff-road_item-item_count'
    feat = feat.merge(tmp,how='left',on=['loc_diff','road_item','item'])
    cols.append(tmp.name)
    
    print('feat loc_diff road_item item fuck done')
            
    feat = feat[ cols ]
    
    return feat

def feat_automl_user_and_recall_type_cate_count(data):
    df = data.copy()
    
    feat = df[ ['index','item','road_item','recall_type','user'] ]
    
    cols = []
    for cate1 in ['user']:
        for cate2 in ['recall_type']:
            for cate3 in ['item','road_item']:
                name3 = f'{cate1}-{cate2}-{cate3}'
                
                feat_tmp = feat.groupby([cate1,cate2,cate3]).size()
                feat_tmp.name = f'{name3}_count'
                feat = feat.merge(feat_tmp,how='left',on=[cate1,cate2,cate3])
                cols.append( name3+'_count' )   
                print(f'feat {cate1} {cate2} {cate3} fuck done')
    
    tmp = feat.groupby(['user','recall_type','road_item','item']).size()
    tmp.name = 'user-recall_type-road_item-item_count'
    feat = feat.merge(tmp,how='left',on=['user','recall_type','road_item','item'])
    cols.append(tmp.name)
    
    print('feat user recall_type road_item item fuck done')            
    
    feat = feat[ cols ]
    return feat

def feat_item_cumcount(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    
    item_time_list = df_train.sort_values('time').groupby('item_id',sort=False)['time'].agg(list)
    
    for i,v in item_time_list.items():
        item_time_list[i] = np.array(v+[1])
    
    df = data.copy()
    
    feat = df[['index','item','query_item_time']]
    
    tmp = feat.set_index('item')
    tmp = tmp.sort_values('query_item_time')
    
    tmp = tmp.groupby(['item']).apply(np.array)
    
    result = np.zeros(df.shape[0])
    
    for i,v in tmp.items():
        time_list = item_time_list[i]
        k = 0
        item_n = v.shape[0]
        for j in range(len(time_list)):
            while k<item_n and v[k,1]<time_list[j]:
                result[int(v[k,0])] = j
                k += 1
    
    feat['item_cumcount'] = result
    
    feat['item_cumrate'] = feat['item_cumcount']/feat['item'].map(df_train['item_id'].value_counts()).fillna(1e-5)
    
    return feat[['item_cumcount','item_cumrate']]

def feat_item_qtime_time_diff(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    

    item_time_list = df_train.sort_values('time').groupby('item_id',sort=False)['time'].agg(list)
    
    for i,v in item_time_list.items():
        item_time_list[i] = np.array([0]+v+[1])
    
    df = data.copy()
    
    feat = df[['index','item','query_item_time']]
    
    tmp = feat.set_index('item')
    tmp = tmp.sort_values('query_item_time')
    
    tmp = tmp.groupby(['item']).apply(np.array)
    
    result_history = np.zeros(df.shape[0])*np.nan
    result_future = np.zeros(df.shape[0])*np.nan
    
    for i,v in tmp.items():
        time_list = item_time_list[i]
        k = 0
        item_n = v.shape[0]
        for j in range(1,len(time_list)):
            while k<item_n and v[k,1]<time_list[j]:
                result_future[int(v[k,0])] = time_list[j]-v[k,1]
                result_history[int(v[k,0])] = v[k,1]-time_list[j-1]
                k += 1
    
    feat['item_qtime_time_diff_history'] = result_history
    feat['item_qtime_time_diff_future'] = result_future
    
    return feat[['item_qtime_time_diff_history','item_qtime_time_diff_future']]

def feat_sim_three_weight_no_clip(data):  
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    loc_weights = {}
    time_weights = {}
    record_weights = {}
    com_item_cnt = {}
    
    item_set = set()
    item_dict_set = {}
    st0 = time.time()
    
    for user, items in user_item_dict.items():
        for item in items:
            item_set.add(item)
            item_dict_set[item] = set()
    
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            loc_weights.setdefault(item, {})
            time_weights.setdefault(item, {})
            record_weights.setdefault(item, {})
            com_item_cnt.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                item_dict_set[ item ].add( relate_item )
                
                t1 = times[loc1]
                t2 = times[loc2]
                loc_weights[item].setdefault(relate_item, 0)
                time_weights[item].setdefault(relate_item, 0)
                record_weights[item].setdefault(relate_item, 0)
                com_item_cnt[item].setdefault(relate_item, 0)
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (0.9**loc_diff)
                    
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (0.9**loc_diff)
                    
                    
                loc_weights[item][relate_item] += loc_weight
                time_weights[item][relate_item] += time_weight
                record_weights[item][relate_item] += len(items)
                com_item_cnt[item][relate_item] += 1

    st1 = time.time()
    print(st1-st0)

    print('start')
    num = feat.shape[0]
    road_item = feat['road_item'].values
    t_item = feat['item'].values
    
    com_item_loc_weights_sum = np.zeros( num, dtype=float )
    com_item_time_weights_sum = np.zeros( num, dtype=float )
    com_item_record_weights_sum = np.zeros( num, dtype=float )
    t_com_item_cnt = np.zeros( num, dtype=float )
    for i in range(num):
        if road_item[i] in item_set:
            if t_item[i] in item_dict_set[ road_item[i] ]:
                com_item_loc_weights_sum[i] = loc_weights[ road_item[i] ][ t_item[i] ] 
                com_item_time_weights_sum[i] = time_weights[ road_item[i] ][ t_item[i] ] 
                com_item_record_weights_sum[i] = record_weights[ road_item[i] ][ t_item[i] ] 
                t_com_item_cnt[i] = com_item_cnt[ road_item[i] ][ t_item[i] ] 
            else:
                com_item_loc_weights_sum[i] = np.nan
                com_item_time_weights_sum[i] = np.nan
                com_item_record_weights_sum[i] = np.nan
                t_com_item_cnt[i] = np.nan
        else:
            com_item_loc_weights_sum[i] = np.nan
            com_item_time_weights_sum[i] = np.nan
            com_item_record_weights_sum[i] = np.nan
            t_com_item_cnt[i] = np.nan

    feat['com_item_loc_weights_sum_no_clip'] = com_item_loc_weights_sum
    feat['com_item_time_weights_sum_no_clip'] = com_item_time_weights_sum
    feat['com_item_record_weights_sum'] = com_item_record_weights_sum
    feat['com_item_cnt'] = t_com_item_cnt
    
    feat['com_item_loc_weights_mean_no_clip'] = feat['com_item_loc_weights_sum_no_clip'] / feat['com_item_cnt']
    feat['com_item_time_weights_mean_no_clip'] = feat['com_item_time_weights_sum_no_clip'] / feat['com_item_cnt']
    feat['com_item_record_weights_mean'] = feat['com_item_record_weights_sum'] / feat['com_item_cnt']

    feat = feat[ ['com_item_loc_weights_sum_no_clip','com_item_time_weights_sum_no_clip',
                  'com_item_loc_weights_mean_no_clip','com_item_time_weights_mean_no_clip', ] ]

    st2 = time.time()
    print(st2-st1)
    return feat

def feat_u2i_road_item_before_and_after_query_time_diff(data):
    df = data.copy()
    feat =  df[['user','road_item_loc','road_item_time','query_item_time']]
    feat_h = feat.loc[feat['road_item_time']<feat['query_item_time']]
    feat_f = feat.loc[feat['road_item_time']>feat['query_item_time']]
    
    feat_h = feat_h.groupby(['user','road_item_loc']).first().reset_index()
    feat_f = feat_f.groupby(['user','road_item_loc']).first().reset_index()
    
    feat_h_group = feat_h.sort_values(['user','road_item_loc']).set_index(['user','road_item_loc']).groupby('user')
    feat_f_group = feat_f.sort_values(['user','road_item_loc']).set_index(['user','road_item_loc']).groupby('user')
    
    feat1 = feat_h_group['road_item_time'].diff(1)
    feat2 = feat_h_group['road_item_time'].diff(-1)
    feat3 = feat_f_group['road_item_time'].diff(1)
    feat4 = feat_f_group['road_item_time'].diff(-1)
    
    feat1.name = 'u2i_road_item_before_query_time_diff_history'
    feat2.name = 'u2i_road_item_before_query_time_diff_future'
    feat3.name = 'u2i_road_item_after_query_time_diff_history'
    feat4.name = 'u2i_road_item_after_query_time_diff_future'
    
    feat = df.merge(pd.concat([feat1,feat2,feat3,feat4],axis=1),how='left',on=['user','road_item_loc'])
    
    cols = ['u2i_road_item_before_query_time_diff_history',
            'u2i_road_item_before_query_time_diff_future',
            'u2i_road_item_after_query_time_diff_history',
            'u2i_road_item_after_query_time_diff_future']
    feat = feat[ cols ]
    return feat

def feat_i2i_cijs_topk_by_loc_new(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        for loc1, item in enumerate(items):
            item_cnt[item] += 1

    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    print('Loading i2i_sim_seq')
    i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')

    print('Getting result')
    topk = 3
    loc_base = 0.9
    
    print(f'Starting {loc_base}')
    result = np.zeros((len(new_keys),4))
    
    for i in range(len(new_keys)):
        key = new_keys[i]
        if key not in i2i_sim_seq.keys():
            result[i,:] = np.nan
            continue
        
        records = i2i_sim_seq[key]
        if len(records)==0:
            print(key)
        
        for record in records:
            loc1, loc2, t1, t2, record_len = record
            if loc1-loc2>0:
                if loc1-loc2==1:
                    result[i,2] += 1
                    result[i,0] += (t1 - t2)
                    
                time_weight = (1 - (t1 - t2) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                    
                loc_diff = loc1-loc2-1
                loc_weight = (loc_base**loc_diff)
                if loc_weight <= 0.2:
                    loc_weight = 0.2        
            else:
                if loc2-loc1==1:
                    result[i,3] += 1
                    result[i,1] += (t2 - t1)
                    
                time_weight = (1 - (t2 - t1) * 100)
                if time_weight<=0.2:
                    time_weight = 0.2
                
                loc_diff =  loc2-loc1-1
                loc_weight =  (loc_base**loc_diff)
                
                if loc_weight <= 0.2:
                    loc_weight = 0.2

    result[:,1]/=(result[:,3]+1e-5)
    result[:,0]/=(result[:,2]+1e-5)
    
    cols = ['history_loc_diff1_com_item_time_mean_new',
            'future_loc_diff1_com_item_time_mean_new',
            'history_loc_diff1_com_item_cnt',
            'future_loc_diff1_com_item_cnt']
    
    result = pd.DataFrame(result,columns=cols,index=new_keys)
    result = result.reset_index()
    result.rename(columns={'index':'new_keys'},inplace=True)
    
    feat = feat.merge(result,how='left',on='new_keys')
    
    print('Finished getting result')
        
    feat = feat[ ['history_loc_diff1_com_item_time_mean_new','future_loc_diff1_com_item_time_mean_new'] ]
    return feat

def feat_items_list_len(data):
    df = data.copy()
    feat = df[ ['index','user','left_items_list','right_items_list','stage'] ]
    def func(s):
        return len(s)
    tdata = feat.groupby('user').first()
    tdata['left_items_list_len'] = tdata['left_items_list'].apply( func )
    tdata['right_items_list_len'] = tdata['right_items_list'].apply( func )
    import pdb
    pdb.set_trace()
    return feat

def feat_item_cnt_in_stage2_mean_max_min_by_user(data):
    if mode=='valid':
        all_train_stage_data = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    else:
        all_train_stage_data = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    item_stage_cnt = all_train_stage_data.groupby(["item_id"])["stage"].value_counts().to_dict()
    feat = data[["user","item", "stage"]]
    feat["head"] = feat.set_index(["item", "stage"]).index
    feat["item_stage_cnt"] = feat["head"].map(item_stage_cnt)
    
    tmp = feat.groupby('user')['item_stage_cnt'].agg(['mean','max','min'])
    tmp.columns = [f'item_cnt_in_stage2_{i}_by_user' for i in tmp.columns]
    
    feat = feat.merge(tmp,how='left',on='user')
    
    return feat[tmp.columns]

def feat_item_seq_sim_cossim_text(data):
    df = data.copy()
    feat = df[ ['left_items_list','right_items_list','item'] ]
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[0]
    
    all_items = np.array(sorted(item_feat.keys()))
    
    item_np = item_np/(np.linalg.norm(item_np,axis=1,keepdims=True)+1e-9)
    
    batch_size = 10000
    n = len(feat)
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    
    
    feat['left_len'] = feat['left_items_list'].apply(len)
    feat_left = feat.sort_values('left_len')
    feat_left_len = feat_left['left_len'].values
    
    feat_left_items_list = feat_left['left_items_list'].values
    feat_left_items = feat_left['item'].values
    
    left_result = np.zeros((len(feat_left),2))
    left_result_len = np.zeros(len(feat_left))
    
    for i in range(batch_num):
        cur_batch_size = len(feat_left_len[i*batch_size:(i+1)*batch_size])
        
        max_len = feat_left_len[i*batch_size:(i+1)*batch_size].max()
        max_len = max(max_len,1)
        left_items = np.zeros((cur_batch_size,max_len),dtype='int32')
        for j,arr in enumerate(feat_left_items_list[i*batch_size:(i+1)*batch_size]):
            left_items[j][:len(arr)] = arr
        
        
        left_result_len[i*batch_size:(i+1)*batch_size] = np.isin(left_items,all_items).sum(axis=1)

        
        vec1 = item_np[left_items]
        vec2 = item_np[feat_left_items[i*batch_size:(i+1)*batch_size]]
        vec2 = vec2.reshape(-1,1,128)
        sim = np.sum(vec1*vec2,axis=-1)
        
        
        left_result[i*batch_size:(i+1)*batch_size,0] = sim.max(axis=1)
        left_result[i*batch_size:(i+1)*batch_size,1] = sim.sum(axis=1)
        
        if i % 10 == 0:
            print('batch num',i)
    
    df_left = pd.DataFrame(left_result,index=feat_left.index,columns=['left_allitem_item_textsim_max','left_allitem_item_textsim_sum'])
    df_left['left_allitem_textsim_len'] = left_result_len
    
    
    
    feat['right_len'] = feat['right_items_list'].apply(len)
    feat_right = feat.sort_values('right_len')
    feat_right_len = feat_right['right_len'].values
    
    feat_right_items_list = feat_right['right_items_list'].values
    feat_right_items = feat_right['item'].values
    
    right_result = np.zeros((len(feat_right),2))
    right_result_len = np.zeros(len(feat_right))
    
    for i in range(batch_num):
        cur_batch_size = len(feat_right_len[i*batch_size:(i+1)*batch_size])
        
        max_len = feat_right_len[i*batch_size:(i+1)*batch_size].max()
        max_len = max(max_len,1)
        right_items = np.zeros((cur_batch_size,max_len),dtype='int32')
        for j,arr in enumerate(feat_right_items_list[i*batch_size:(i+1)*batch_size]):
            right_items[j][:len(arr)] = arr
        
        
        right_result_len[i*batch_size:(i+1)*batch_size] = np.isin(right_items,all_items).sum(axis=1)

        
        vec1 = item_np[right_items]
        vec2 = item_np[feat_right_items[i*batch_size:(i+1)*batch_size]]
        vec2 = vec2.reshape(-1,1,128)
        sim = np.sum(vec1*vec2,axis=-1)
        
        
        right_result[i*batch_size:(i+1)*batch_size,0] = sim.max(axis=1)
        right_result[i*batch_size:(i+1)*batch_size,1] = sim.sum(axis=1)
        
        if i % 10 == 0:
            print('batch num',i)
    df_right = pd.DataFrame(right_result,index=feat_right.index,columns=['right_allitem_item_textsim_max','right_allitem_item_textsim_sum'])
    df_right['right_allitem_textsim_len'] = right_result_len
    
    
    df_left = df_left.sort_index()
    df_right = df_right.sort_index()
    
    feat = pd.concat([df_left,df_right],axis=1)
    
    feat['allitem_item_textsim_max'] = feat[['left_allitem_item_textsim_max','right_allitem_item_textsim_max']].max(axis=1)
    feat['allitem_item_textsim_sum'] = feat[['left_allitem_item_textsim_sum','right_allitem_item_textsim_sum']].sum(axis=1)
    feat['allitem_item_textsim_len'] = feat[['left_allitem_textsim_len','right_allitem_textsim_len']].sum(axis=1)
    feat['allitem_item_textsim_mean'] = feat['allitem_item_textsim_sum']/(feat['allitem_item_textsim_len']+1e-9)
    
    
    return feat[['allitem_item_textsim_max','allitem_item_textsim_mean']]
def feat_item_seq_sim_cossim_image(data):
    df = data.copy()
    feat = df[ ['left_items_list','right_items_list','item'] ]
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[1]
    
    all_items = np.array(sorted(item_feat.keys()))
    
    item_np = item_np/(np.linalg.norm(item_np,axis=1,keepdims=True)+1e-9)
    
    batch_size = 10000
    n = len(feat)
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    
    
    feat['left_len'] = feat['left_items_list'].apply(len)
    feat_left = feat.sort_values('left_len')
    feat_left_len = feat_left['left_len'].values
    
    feat_left_items_list = feat_left['left_items_list'].values
    feat_left_items = feat_left['item'].values
    
    left_result = np.zeros((len(feat_left),2))
    left_result_len = np.zeros(len(feat_left))
    
    for i in range(batch_num):
        cur_batch_size = len(feat_left_len[i*batch_size:(i+1)*batch_size])
        
        max_len = feat_left_len[i*batch_size:(i+1)*batch_size].max()
        max_len = max(max_len,1)
        left_items = np.zeros((cur_batch_size,max_len),dtype='int32')
        for j,arr in enumerate(feat_left_items_list[i*batch_size:(i+1)*batch_size]):
            left_items[j][:len(arr)] = arr
        
        
        left_result_len[i*batch_size:(i+1)*batch_size] = np.isin(left_items,all_items).sum(axis=1)

        
        vec1 = item_np[left_items]
        vec2 = item_np[feat_left_items[i*batch_size:(i+1)*batch_size]]
        vec2 = vec2.reshape(-1,1,128)
        sim = np.sum(vec1*vec2,axis=-1)
        
        
        left_result[i*batch_size:(i+1)*batch_size,0] = sim.max(axis=1)
        left_result[i*batch_size:(i+1)*batch_size,1] = sim.sum(axis=1)
        
        if i % 10 == 0:
            print('batch num',i)
    
    df_left = pd.DataFrame(left_result,index=feat_left.index,columns=['left_allitem_item_imagesim_max','left_allitem_item_imagesim_sum'])
    df_left['left_allitem_imagesim_len'] = left_result_len
    
    
    
    feat['right_len'] = feat['right_items_list'].apply(len)
    feat_right = feat.sort_values('right_len')
    feat_right_len = feat_right['right_len'].values
    
    feat_right_items_list = feat_right['right_items_list'].values
    feat_right_items = feat_right['item'].values
    
    right_result = np.zeros((len(feat_right),2))
    right_result_len = np.zeros(len(feat_right))
    
    for i in range(batch_num):
        cur_batch_size = len(feat_right_len[i*batch_size:(i+1)*batch_size])
        
        max_len = feat_right_len[i*batch_size:(i+1)*batch_size].max()
        max_len = max(max_len,1)
        right_items = np.zeros((cur_batch_size,max_len),dtype='int32')
        for j,arr in enumerate(feat_right_items_list[i*batch_size:(i+1)*batch_size]):
            right_items[j][:len(arr)] = arr
        
        
        right_result_len[i*batch_size:(i+1)*batch_size] = np.isin(right_items,all_items).sum(axis=1)

        
        vec1 = item_np[right_items]
        vec2 = item_np[feat_right_items[i*batch_size:(i+1)*batch_size]]
        vec2 = vec2.reshape(-1,1,128)
        sim = np.sum(vec1*vec2,axis=-1)
        
        
        right_result[i*batch_size:(i+1)*batch_size,0] = sim.max(axis=1)
        right_result[i*batch_size:(i+1)*batch_size,1] = sim.sum(axis=1)
        
        if i % 10 == 0:
            print('batch num',i)
    df_right = pd.DataFrame(right_result,index=feat_right.index,columns=['right_allitem_item_imagesim_max','right_allitem_item_imagesim_sum'])
    df_right['right_allitem_imagesim_len'] = right_result_len
    
    
    df_left = df_left.sort_index()
    df_right = df_right.sort_index()
    
    feat = pd.concat([df_left,df_right],axis=1)
    
    feat['allitem_item_imagesim_max'] = feat[['left_allitem_item_imagesim_max','right_allitem_item_imagesim_max']].max(axis=1)
    feat['allitem_item_imagesim_sum'] = feat[['left_allitem_item_imagesim_sum','right_allitem_item_imagesim_sum']].sum(axis=1)
    feat['allitem_item_imagesim_len'] = feat[['left_allitem_imagesim_len','right_allitem_imagesim_len']].sum(axis=1)
    feat['allitem_item_imagesim_mean'] = feat['allitem_item_imagesim_sum']/(feat['allitem_item_imagesim_len']+1e-9)
    
    
    return feat[['allitem_item_imagesim_max','allitem_item_imagesim_mean']]
def feat_i2i_sim_on_hist_seq(data):
    # get i2i similarities dict
    # 没用
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    sim_item = {}
    item_cnt = defaultdict(int)
    com_item_cnt = {}
    
    item_set = set()
    item_dict_set = {}
    
    st0 = time.time()
    for user, items in user_item_dict.items():
        for item in items:
            item_set.add(item)
            item_dict_set[item] = set()
            
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            com_item_cnt.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                item_dict_set[ item ].add( relate_item )
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                com_item_cnt[item].setdefault(relate_item, 0)
                
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (0.9**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2

                    sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items))
                          
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (0.9**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                    sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items)) 
                com_item_cnt[item][relate_item] += 1.0

    print("compute i2i sim end.")
    max_i2i_sim_arr = np.zeros(len(data))
    mean_i2i_sim_arr = np.zeros(len(data))

    # 还可以做过去N次点击，过去N时间内的统计
    for i, (left_seq, right_seq, item) in enumerate(zip(data["left_items_list"].values, data["right_items_list"].values, data["item"].values)):
        if i % 100000 == 0:
            print("{} in length {}".format(i, len(data)))
        seq_i2i_sim = []
        for h_item in left_seq + right_seq:
            sim_item[h_item].setdefault(item, 0)
            seq_i2i_sim.append(sim_item[h_item][item])
        max_i2i_sim_arr[i] = max(seq_i2i_sim) if len(left_seq) > 0 else np.nan
        mean_i2i_sim_arr[i] = sum(seq_i2i_sim) / len(left_seq) if len(left_seq) > 0 else np.nan

    feat = data[["item"]]
    feat["max_i2i_sim_arr"] = max_i2i_sim_arr
    feat["mean_i2i_sim_arr"] = mean_i2i_sim_arr
    
    return feat[[
        "max_i2i_sim_arr", "mean_i2i_sim_arr"
    ]]


def feat_item_seq_sim_cossim_text(data):
    df = data.copy()
    feat = df[ ['left_items_list','right_items_list','item'] ]
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[0]
    
    all_items = np.array(sorted(item_feat.keys()))
    
    item_np = item_np/(np.linalg.norm(item_np,axis=1,keepdims=True)+1e-9)
    
    batch_size = 30000
    n = len(feat)
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    
    feat['left_len'] = feat['left_items_list'].apply(len)
    feat_left = feat.sort_values('left_len')
    feat_left_len = feat_left['left_len'].values
    
    feat_left_items_list = feat_left['left_items_list'].values
    feat_left_items = feat_left['item'].values
    
    left_result = np.zeros((len(feat_left),2))
    left_result_len = np.zeros(len(feat_left))
    
    len_max_nums = 300
    
    for i in range(batch_num):
        cur_batch_size = len(feat_left_len[i*batch_size:(i+1)*batch_size])
        
        max_len = feat_left_len[i*batch_size:(i+1)*batch_size].max()
        max_len = min(max(max_len,1),len_max_nums)
        left_items = np.zeros((cur_batch_size,max_len),dtype='int32')
        for j,arr in enumerate(feat_left_items_list[i*batch_size:(i+1)*batch_size]):
            arr = arr[:len_max_nums]
            left_items[j][:len(arr)] = arr
        
        
        left_result_len[i*batch_size:(i+1)*batch_size] = np.isin(left_items,all_items).sum(axis=1)

        
        vec1 = item_np[left_items]
        vec2 = item_np[feat_left_items[i*batch_size:(i+1)*batch_size]]
        vec2 = vec2.reshape(-1,1,128)
        sim = np.sum(vec1*vec2,axis=-1)
        
        
        left_result[i*batch_size:(i+1)*batch_size,0] = sim.max(axis=1)
        left_result[i*batch_size:(i+1)*batch_size,1] = sim.sum(axis=1)
        
        if i % 10 == 0:
            print('batch num',i)
    
    df_left = pd.DataFrame(left_result,index=feat_left.index,columns=['left_allitem_item_textsim_max','left_allitem_item_textsim_sum'])
    df_left['left_allitem_textsim_len'] = left_result_len
    
    
    
    feat['right_len'] = feat['right_items_list'].apply(len)
    feat_right = feat.sort_values('right_len')
    feat_right_len = feat_right['right_len'].values
    
    feat_right_items_list = feat_right['right_items_list'].values
    feat_right_items = feat_right['item'].values
    
    right_result = np.zeros((len(feat_right),2))
    right_result_len = np.zeros(len(feat_right))
    
    len_max_nums = 80
    
    for i in range(batch_num):
        cur_batch_size = len(feat_right_len[i*batch_size:(i+1)*batch_size])
        
        max_len = feat_right_len[i*batch_size:(i+1)*batch_size].max()
        max_len = min(max(max_len,1),len_max_nums)
        right_items = np.zeros((cur_batch_size,max_len),dtype='int32')
        for j,arr in enumerate(feat_right_items_list[i*batch_size:(i+1)*batch_size]):
            arr = arr[:len_max_nums]
            right_items[j][:len(arr)] = arr
        
        
        right_result_len[i*batch_size:(i+1)*batch_size] = np.isin(right_items,all_items).sum(axis=1)

        
        vec1 = item_np[right_items]
        vec2 = item_np[feat_right_items[i*batch_size:(i+1)*batch_size]]
        vec2 = vec2.reshape(-1,1,128)
        sim = np.sum(vec1*vec2,axis=-1)
        
        
        right_result[i*batch_size:(i+1)*batch_size,0] = sim.max(axis=1)
        right_result[i*batch_size:(i+1)*batch_size,1] = sim.sum(axis=1)
        
        if i % 10 == 0:
            print('batch num',i)
    df_right = pd.DataFrame(right_result,index=feat_right.index,columns=['right_allitem_item_textsim_max','right_allitem_item_textsim_sum'])
    df_right['right_allitem_textsim_len'] = right_result_len
    
    
    df_left = df_left.sort_index()
    df_right = df_right.sort_index()
    
    feat = pd.concat([df_left,df_right],axis=1)
    
    feat['allitem_item_textsim_max'] = feat[['left_allitem_item_textsim_max','right_allitem_item_textsim_max']].max(axis=1)
    feat['allitem_item_textsim_sum'] = feat[['left_allitem_item_textsim_sum','right_allitem_item_textsim_sum']].sum(axis=1)
    feat['allitem_item_textsim_len'] = feat[['left_allitem_textsim_len','right_allitem_textsim_len']].sum(axis=1)
    feat['allitem_item_textsim_mean'] = feat['allitem_item_textsim_sum']/(feat['allitem_item_textsim_len']+1e-9)
    
    
    return feat[['allitem_item_textsim_max','allitem_item_textsim_mean']]

def feat_item_seq_sim_cossim_image(data):
    df = data.copy()
    feat = df[ ['left_items_list','right_items_list','item'] ]
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[1]
    
    all_items = np.array(sorted(item_feat.keys()))
    
    item_np = item_np/(np.linalg.norm(item_np,axis=1,keepdims=True)+1e-9)
    
    batch_size = 30000
    n = len(feat)
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    
    
    feat['left_len'] = feat['left_items_list'].apply(len)
    feat_left = feat.sort_values('left_len')
    feat_left_len = feat_left['left_len'].values
    
    feat_left_items_list = feat_left['left_items_list'].values
    feat_left_items = feat_left['item'].values
    
    left_result = np.zeros((len(feat_left),2))
    left_result_len = np.zeros(len(feat_left))
    
    len_max_nums = 300
    
    for i in range(batch_num):
        cur_batch_size = len(feat_left_len[i*batch_size:(i+1)*batch_size])
        
        max_len = feat_left_len[i*batch_size:(i+1)*batch_size].max()
        max_len = min(max(max_len,1),len_max_nums)
        left_items = np.zeros((cur_batch_size,max_len),dtype='int32')
        for j,arr in enumerate(feat_left_items_list[i*batch_size:(i+1)*batch_size]):
            arr = arr[:len_max_nums]
            left_items[j][:len(arr)] = arr
        
        
        left_result_len[i*batch_size:(i+1)*batch_size] = np.isin(left_items,all_items).sum(axis=1)

        
        vec1 = item_np[left_items]
        vec2 = item_np[feat_left_items[i*batch_size:(i+1)*batch_size]]
        vec2 = vec2.reshape(-1,1,128)
        sim = np.sum(vec1*vec2,axis=-1)
        
        
        left_result[i*batch_size:(i+1)*batch_size,0] = sim.max(axis=1)
        left_result[i*batch_size:(i+1)*batch_size,1] = sim.sum(axis=1)
        
        if i % 10 == 0:
            print('batch num',i)
    
    df_left = pd.DataFrame(left_result,index=feat_left.index,columns=['left_allitem_item_imagesim_max','left_allitem_item_imagesim_sum'])
    df_left['left_allitem_imagesim_len'] = left_result_len
    
    
    
    feat['right_len'] = feat['right_items_list'].apply(len)
    feat_right = feat.sort_values('right_len')
    feat_right_len = feat_right['right_len'].values
    
    feat_right_items_list = feat_right['right_items_list'].values
    feat_right_items = feat_right['item'].values
    
    right_result = np.zeros((len(feat_right),2))
    right_result_len = np.zeros(len(feat_right))
    
    len_max_nums = 80
    
    for i in range(batch_num):
        cur_batch_size = len(feat_right_len[i*batch_size:(i+1)*batch_size])
        
        max_len = feat_right_len[i*batch_size:(i+1)*batch_size].max()
        max_len = min(max(max_len,1),len_max_nums)
        right_items = np.zeros((cur_batch_size,max_len),dtype='int32')
        for j,arr in enumerate(feat_right_items_list[i*batch_size:(i+1)*batch_size]):
            arr = arr[:len_max_nums]
            right_items[j][:len(arr)] = arr
        
        right_result_len[i*batch_size:(i+1)*batch_size] = np.isin(right_items,all_items).sum(axis=1)

        
        vec1 = item_np[right_items]
        vec2 = item_np[feat_right_items[i*batch_size:(i+1)*batch_size]]
        vec2 = vec2.reshape(-1,1,128)
        sim = np.sum(vec1*vec2,axis=-1)
        
        
        right_result[i*batch_size:(i+1)*batch_size,0] = sim.max(axis=1)
        right_result[i*batch_size:(i+1)*batch_size,1] = sim.sum(axis=1)
        
        if i % 10 == 0:
            print('batch num',i)
    df_right = pd.DataFrame(right_result,index=feat_right.index,columns=['right_allitem_item_imagesim_max','right_allitem_item_imagesim_sum'])
    df_right['right_allitem_imagesim_len'] = right_result_len
    
    
    df_left = df_left.sort_index()
    df_right = df_right.sort_index()
    
    feat = pd.concat([df_left,df_right],axis=1)
    
    feat['allitem_item_imagesim_max'] = feat[['left_allitem_item_imagesim_max','right_allitem_item_imagesim_max']].max(axis=1)
    feat['allitem_item_imagesim_sum'] = feat[['left_allitem_item_imagesim_sum','right_allitem_item_imagesim_sum']].sum(axis=1)
    feat['allitem_item_imagesim_len'] = feat[['left_allitem_imagesim_len','right_allitem_imagesim_len']].sum(axis=1)
    feat['allitem_item_imagesim_mean'] = feat['allitem_item_imagesim_sum']/(feat['allitem_item_imagesim_len']+1e-9)
    
    
    return feat[['allitem_item_imagesim_max','allitem_item_imagesim_mean']]

def feat_i2i_sim_on_hist_seq(data):
    # get i2i similarities dict
    # 没用
    df = data.copy()
    feat = df[ ['index','road_item','item'] ]
    
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    sim_item = {}
    item_cnt = defaultdict(int)
    com_item_cnt = {}
    
    item_set = set()
    item_dict_set = {}
    
    st0 = time.time()
    for user, items in user_item_dict.items():
        for item in items:
            item_set.add(item)
            item_dict_set[item] = set()
            
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            com_item_cnt.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                item_dict_set[ item ].add( relate_item )
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                com_item_cnt[item].setdefault(relate_item, 0)
                
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (0.9**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2

                    sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items))
                          
                else:
                    time_weight = (1 - (t2 - t1) * 100)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (0.9**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                    sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1 + len(items)) 
                com_item_cnt[item][relate_item] += 1.0

    print("compute i2i sim end.")
    max_i2i_sim_arr = np.zeros(len(data))
    mean_i2i_sim_arr = np.zeros(len(data))
    
    left_len_max_nums = 300
    right_len_max_nums = 80
    
    # 还可以做过去N次点击，过去N时间内的统计
    for i, (left_seq, right_seq, item) in enumerate(zip(data["left_items_list"].values, data["right_items_list"].values, data["item"].values)):
        if i % 100000 == 0:
            print("{} in length {}".format(i, len(data)))
        seq_i2i_sim = []
        left_seq = left_seq[:left_len_max_nums]
        right_seq = right_seq[:right_len_max_nums]
        left_right_seq = left_seq+right_seq
        for h_item in left_right_seq:
            sim_item[h_item].setdefault(item, 0)
            seq_i2i_sim.append(sim_item[h_item][item])
        
        max_i2i_sim_arr[i] = max(seq_i2i_sim) if len(left_right_seq) > 0 else np.nan
        mean_i2i_sim_arr[i] = sum(seq_i2i_sim) / len(left_right_seq) if len(left_right_seq) > 0 else np.nan

    feat = data[["item"]]
    
    feat["max_i2i_sim_arr"] = max_i2i_sim_arr
    feat["mean_i2i_sim_arr"] = mean_i2i_sim_arr
    
    return feat[[
        "max_i2i_sim_arr", "mean_i2i_sim_arr"
    ]]


def feat_item_max_sim_weight_loc_weight_time_weight_rank_weight(data):
    df = data.copy()
    df = df[ ['user','item','sim_weight','loc_weight','time_weight','rank_weight','index'] ]
    feat = df[ ['index','user','item'] ]
    df = df.groupby( ['user','item'] )[ ['sim_weight','loc_weight','time_weight','rank_weight'] ].agg( ['max'] ).reset_index()
    cols = [ f'item_{j}_{i}' for i in ['sim_weight','loc_weight','time_weight','rank_weight'] for j in ['max'] ]
    df.columns = [ 'user','item' ]+ cols
    feat = pd.merge( feat, df, on=['user','item'], how='left')
    feat = feat[ cols ] 
    return feat

def feat_different_type_road_score_max(data):  
    df = data.copy()
    feat = df[ ['user','item','index','sim_weight','recall_type'] ]
    feat['i2i_score'] = feat['sim_weight']
    feat['blend_score'] = feat['sim_weight']
    feat['i2i2i_score'] = feat['sim_weight']
    feat.loc[ feat['recall_type']!=0 , 'i2i_score'] = np.nan
    feat.loc[ feat['recall_type']!=1 , 'blend_score'] = np.nan
    feat.loc[ feat['recall_type']!=2 , 'i2i2i_score'] = np.nan

    df = feat[ ['index','user','item','i2i_score','blend_score','i2i2i_score'] ]
    df = df.groupby( ['user','item'] )[ ['i2i_score','blend_score','i2i2i_score'] ].agg( ['max'] ).reset_index()
    df.columns = ['user','item'] + [ f'{i}_{j}' for i in ['i2i_score','blend_score','i2i2i_score'] for j in ['max'] ]
    feat = pd.merge( feat, df, on=['user','item'], how='left')
    feat = feat[ ['i2i_score_max','blend_score_max','i2i2i_score_max',] ]
    return feat

def feat_different_type_road_score_max_by_item(data):  
    df = data.copy()
    feat = df[ ['item','index','sim_weight','recall_type'] ]
    
    cols = ['i2i_score','blend_score','i2i2i_score']#,'i2iw10_score','i2i2b_score']
    for i in range(len(cols)):
        feat[cols[i]] = feat['sim_weight']
        feat.loc[ feat['recall_type']!=i,cols[i] ] = np.nan
    
    
    df = feat[ ['index','item','i2i_score','blend_score','i2i2i_score'] ]
    df = df.groupby( ['item'] )[ ['i2i_score','blend_score','i2i2i_score'] ].agg( ['max'] ).reset_index()
    df.columns = ['item'] + [ f'{i}_{j}_by_item' for i in ['i2i_score','blend_score','i2i2i_score'] for j in ['max'] ]
    feat = pd.merge( feat, df, on=['item'], how='left')
    feat = feat[ ['i2i_score_max_by_item','blend_score_max_by_item','i2i2i_score_max_by_item',] ]
    
    return feat

def feat_item_sum_mean_max_i2i2i_weight(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))

    df = data.copy()
    feat = df[ ['index','road_item','item','user'] ]
    print('Loading i2i2i_sim_seq')
    i2i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')
    print('Getting result')
    
    result = np.zeros(len(new_keys))
    
    item_cnt = df_train['item_id'].value_counts().to_dict()
    
    for i in range(len(new_keys)):
        key = new_keys[i]
        if key not in i2i2i_sim_seq.keys():
            result[i] = np.nan
            continue  
        records = i2i2i_sim_seq[key]
        
        if len(records)==0:
            print(key)
        for record in records:
            item,score1_1,score1_2,score2_1,score2_2 = record
            result[i] += score1_1*score1_2
        
    print('Finished getting result')
    result = pd.DataFrame(result,index=new_keys,columns=['i2i2i_score1_sum'])
    result = result.reset_index()
    result.rename(columns={'index':'new_keys'},inplace=True)
    
    feat = feat.merge(result,how='left',on='new_keys')
    
    tmp = feat.groupby( ['user','item'] )[ ['i2i2i_score1_sum'] ].agg( ['sum','mean','max'] ).reset_index()
    cols = [ f'item_{j}_{i}' for i in ['i2i2i_weight'] for j in ['sum','mean','max'] ]
    tmp.columns = [ 'user','item' ]+ cols
    feat = pd.merge( feat, tmp, on=['user','item'], how='left')
    feat = feat[ cols ] 
    
    return feat

def feat_item_sum_mean_max_sim_weight_loc_weight_time_weight_rank_weight_by_item(data):
    df = data.copy()
    df = df[ ['user','item','sim_weight','loc_weight','time_weight','rank_weight','index'] ]
    feat = df[ ['index','user','item'] ]
    df = df.groupby( ['item'] )[ ['sim_weight','loc_weight','time_weight','rank_weight'] ].agg( ['sum','mean','max'] ).reset_index()
    cols = [ f'item_{j}_{i}_by_item' for i in ['sim_weight','loc_weight','time_weight','rank_weight'] for j in ['sum','mean','max'] ]
    df.columns = [ 'item' ]+ cols
    feat = pd.merge( feat, df, on=['item'], how='left')
    feat = feat[ cols ] 
    return feat



def compare_data(data):
    df = data.copy()
    feat = df[ ['index','road_item','item','loc_weight','time_weight'] ]
    feat1 = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_item_cumcount_new_{mode}_{cur_stage}.pkl') )
    feat2 = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_item_cumcount_{mode}_{cur_stage}.pkl') )
    feat3 = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_item_qtime_time_diff_new_{mode}_{cur_stage}.pkl') )
    feat4 = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_item_qtime_time_diff_{mode}_{cur_stage}.pkl') )
    #feat5 = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_automl_user_and_recall_type_cate_count_new_{mode}_{cur_stage}.pkl') )
    #feat6 = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_automl_user_and_recall_type_cate_count_{mode}_{cur_stage}.pkl') )

    import pdb
    pdb.set_trace()
    print(1)

def feat_test(data):
    df = data.copy()
    categories = [ 'user','item','road_item','road_item_loc',
                  'query_item_loc','recall_type']
    feat = df[ ['index']+categories ]
    
    feat['loc_diff'] = df['query_item_loc']-df['road_item_loc']
    
    categories += ['loc_diff']
    
    n = len(categories)
    cols = []
    for a in range(n):
        cate1 = categories[a]
        feat[cate1+'_count_'] = feat[cate1].map( feat[cate1].value_counts() )
        cols.append( cate1+'_count_' )
        for b in range(a+1,n):
            cate2 = categories[b]
            name2 = f'{cate1}_{cate2}'
            feat[name2] = feat[cate1].astype('str') + '-' + feat[cate2].astype('str')
            feat[name2+'_count_'] = feat[name2].map( feat[name2].value_counts() )
            cols.append( name2+'_count_' )   
            for c in range(b+1,n):
                cate3 = categories[c]
                name3 = f'{cate1}_{cate2}_{cate3}'
                feat[name3] = feat[cate1].astype('str') + '-' + feat[cate2].astype('str') + '-' + feat[cate3].astype('str')
                feat[name3+'_count_'] = feat[name3].map( feat[name3].value_counts() )
                cols.append( name3+'_count_' )
    feat = feat[ cols ]
    return feat

def feat_item_sum_mean_max_i2i2i_weight_by_item(data):
    if mode == 'valid':
        df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    elif mode == 'test':
        df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))

    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))

    df = data.copy()
    feat = df[ ['index','road_item','item','user'] ]
    print('Loading i2i2i_sim_seq')
    i2i2i_sim_seq = utils.load_pickle( feat_dir + f'{used_recall_source}/' +  (f'feat_i2i2i_seq_{mode}_{cur_stage}.pkl') )
    print('Finished i2i2i_sim_seq')

    print('Creat new key')
    vals = feat[ ['road_item', 'item'] ].values
    new_keys = []
    for val in vals:
        new_keys.append( (val[0], val[1]) )
    feat['new_keys'] = new_keys
    new_keys = sorted( list( set(new_keys) ) )
    print('Finished new key')
    print('Getting result')
    
    result = np.zeros(len(new_keys))
    
    item_cnt = df_train['item_id'].value_counts().to_dict()
    
    for i in range(len(new_keys)):
        key = new_keys[i]
        if key not in i2i2i_sim_seq.keys():
            result[i] = np.nan
            continue  
        records = i2i2i_sim_seq[key]
        
        if len(records)==0:
            print(key)
        for record in records:
            item,score1_1,score1_2,score2_1,score2_2 = record
            result[i] += score1_1*score1_2
        
    print('Finished getting result')
    result = pd.DataFrame(result,index=new_keys,columns=['i2i2i_score1_sum'])
    result = result.reset_index()
    result.rename(columns={'index':'new_keys'},inplace=True)
    
    feat = feat.merge(result,how='left',on='new_keys')
    
    tmp = feat.groupby( ['item'] )[ ['i2i2i_score1_sum'] ].agg( ['sum','mean','max'] ).reset_index()
    cols = [ f'item_{j}_{i}_by_item' for i in ['i2i2i_weight'] for j in ['sum','mean','max'] ]
    tmp.columns = [ 'item' ]+ cols
    feat = pd.merge( feat, tmp, on=['item'], how='left')
    feat = feat[ cols ] 
    
    return feat

def feat_item_sum_mean_max_t2t_weight_by_item(data):
    df = data.copy()
    feat = df[ ['index','road_item','item','user'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[0]
    
    item_l2 = np.linalg.norm(item_np,axis=1)
    
    n = feat.shape[0]
    
    result = np.zeros(n)
    
    batch_size = 100000
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    for i in range(batch_num):
        result[i*batch_size:(i+1)*batch_size] = np.multiply(item_np[feat['road_item'][i*batch_size:(i+1)*batch_size],:],item_np[feat['item'][i*batch_size:(i+1)*batch_size],:]).sum(axis=1)
    
    feat['road_item_text_dot'] = result
    feat.loc[(~feat['item'].isin(item_feat.keys()))|(~feat['road_item'].isin(item_feat.keys())),'road_item_text_dot'] = np.nan
    
    tmp = feat.groupby( ['item'] )[ ['road_item_text_dot'] ].agg( ['sum','mean','max'] ).reset_index()
    cols = [ f'item_{j}_{i}_by_item' for i in ['t2t_weight'] for j in ['sum','mean','max'] ]
    tmp.columns = [ 'item' ]+ cols
    feat = pd.merge( feat, tmp, on=['item'], how='left')
    feat = feat[ cols ] 
    return feat

def feat_item_sum_mean_max_p2p_weight_by_item(data):
    df = data.copy()
    feat = df[ ['index','road_item','item','user'] ]
    item_text = {}
    item_feat = utils.load_pickle(item_feat_pkl)
    
    item_n = 120000
    item_np = np.zeros((item_n,128))
    
    for k,v in item_feat.items():
        item_np[k,:] = v[1]
    
    item_l2 = np.linalg.norm(item_np,axis=1)
    
    n = feat.shape[0]
    
    result = np.zeros(n)
    
    batch_size = 100000
    batch_num = n//batch_size if n%batch_size==0 else n//batch_size+1
    for i in range(batch_num):
        result[i*batch_size:(i+1)*batch_size] = np.multiply(item_np[feat['road_item'][i*batch_size:(i+1)*batch_size],:],item_np[feat['item'][i*batch_size:(i+1)*batch_size],:]).sum(axis=1)
    
    feat['road_item_image_dot'] = result
    feat.loc[(~feat['item'].isin(item_feat.keys()))|(~feat['road_item'].isin(item_feat.keys())),'road_item_image_dot'] = np.nan
    
    tmp = feat.groupby( ['item'] )[ ['road_item_image_dot'] ].agg( ['sum','mean','max'] ).reset_index()
    cols = [ f'item_{j}_{i}_by_item' for i in ['p2p_weight'] for j in ['sum','mean','max'] ]
    tmp.columns = [ 'item' ]+ cols
    feat = pd.merge( feat, tmp, on=['item'], how='left')
    feat = feat[ cols ] 
    return feat

if __name__ == '__main__':
    
    good_funcs = [ feat_item_sum_mean_sim_weight_loc_weight_time_weight_rank_weight,
                feat_sum_sim_loc_time_weight,
                feat_road_item_text_cossim, feat_road_item_text_eulasim,
                feat_sim_base, feat_sim_three_weight, #time_weights, loc_weights clip problem
                feat_different_type_road_score_sum_mean, feat_u2i_road_item_time_diff, #split history,future -> 4 cols
                feat_road_item_text_dot, feat_road_item_text_norm2, 
                feat_time_window_cate_count, 
                feat_i2i_seq, feat_automl_recall_type_cate_count, feat_automl_loc_diff_cate_count, 
                feat_automl_user_and_recall_type_cate_count, feat_i2i_cijs_topk_by_loc, feat_i2i_cijs_median_mean_topk, # feat_i2i_cijs_topk_by_loc divide problem
                feat_different_type_road_score_sum_mean_by_item, item_cnt_in_stage2,
                feat_item_cnt_in_different_stage, feat_different_type_road_score_mean_by_road_item, 
                feat_i2i2i_seq, feat_i2i2i_sim,
                feat_item_stage_nunique, feat_item_qtime_time_diff, feat_item_cumcount, feat_road_time_bins_cate_cnt,
                feat_u2i_road_item_before_and_after_query_time_diff, 
                feat_item_cnt_in_stage2_mean_max_min_by_user, 
                feat_item_seq_sim_cossim_text,
                feat_item_seq_sim_cossim_image,
                feat_i2i_sim_on_hist_seq,
                ]   
    print('running features num: ', len(good_funcs) )
    
    funcs = good_funcs
    if not os.path.exists(feat_dir + f'{used_recall_source}/'):
        os.makedirs(feat_dir + f'{used_recall_source}/')

    data = utils.load_pickle( lgb_base_pkl.format(used_recall_source,mode,cur_stage) )
    data = data.reset_index()

    print(data.shape, 'now mode : ', mode, 'cur_stage: ', cur_stage )
    for func in funcs:
        t1 = time.time()
        feat = func(data)
        if type(feat) == pd.DataFrame:
            for col in feat.columns:
                feat[col] = utils.downcast(feat[col])
        feat_path = (func.__name__+'_{}_{}.pkl').format(mode,cur_stage)
        t2 = time.time()
        print('do feature {} use time {} s'.format( func.__name__, t2-t1 ))
        utils.dump_pickle(feat, feat_dir + f'{used_recall_source}/' + feat_path)
        t3 = time.time()
        print('save feature {} use time {} s'.format( func.__name__, t3-t2 ))
    
    '''
    look
    '''