# -*- coding: utf-8 -*-

import numpy as np
import utils
from constants import *
import ndcg_tools
import pandas as pd
from collections import defaultdict
import math
import gc
import sys
cur_stage = CUR_STAGE

recall_left_max_road_num = 30
recall_left_max_num_each_road = 300
recall_right_max_road_num = 30
recall_right_max_num_each_road = 300

big_or_small_or_history = 'big' # big small history
if len(sys.argv)>0:
    big_or_small_or_history = sys.argv[1]
print('big_or_small_or_history: ', big_or_small_or_history)
#'big'
#'his30-fut1'
#'his30-fut4'
i2i_sim_limit = 250
b2b_sim_limit = 350
i2i2i_sim_limit = 100
i2i2i_new_sim_limit = 100
b2b2b_sim_limit = 100
i2i2b_i_sim_limit = 100
i2i2b_b_sim_limit = 100
b2b2i_i_sim_limit = 100
b2b2i_b_sim_limit = 100

if big_or_small_or_history == 'big':
    i2i_sim_limit = 250
    b2b_sim_limit = 350
    i2i2i_sim_limit = 100
    i2i2i_new_sim_limit = 100
    b2b2b_sim_limit = 100
    i2i2b_i_sim_limit = 100
    i2i2b_b_sim_limit = 100
    b2b2i_i_sim_limit = 100
    b2b2i_b_sim_limit = 100
elif big_or_small_or_history == 'small':
    i2i_sim_limit = 100
    b2b_sim_limit = 100
    i2i2i_sim_limit = 50
    i2i2i_new_sim_limit = 50
    b2b2b_sim_limit = 50
    i2i2b_i_sim_limit = 50
    i2i2b_b_sim_limit = 50
    b2b2i_i_sim_limit = 50
    b2b2i_b_sim_limit = 50
elif big_or_small_or_history == 'history':
    i2i_sim_limit = 250
    b2b_sim_limit = 350
    i2i2i_sim_limit = 100
    i2i2i_new_sim_limit = 100
    b2b2b_sim_limit = 100
    i2i2b_i_sim_limit = 100
    i2i2b_b_sim_limit = 100
    b2b2i_i_sim_limit = 100
    b2b2i_b_sim_limit = 100
elif big_or_small_or_history in ['his30-fut0','his30-fut1','his30-fut4','his30-fut8'] :
    i2i_sim_limit = 250
    b2b_sim_limit = 350
    i2i2i_sim_limit = 100
    i2i2i_new_sim_limit = 100
    b2b2b_sim_limit = 100
    i2i2b_i_sim_limit = 100
    i2i2b_b_sim_limit = 100
    b2b2i_i_sim_limit = 100
    b2b2i_b_sim_limit = 100
else:
    print(1/0)

mode = cur_mode
cal_sim = True
get_sum = False #同一种召回方法的话同一个item是否合并。


def recommend(sim_item_corr, user_item_dict, user_time_dict, user_id, qtime, loc_coff=0.7, recall_type=None):
    interacted_items = user_item_dict[user_id] 
    interacted_times = user_time_dict[user_id]
    qtime_loc = 0
    while qtime_loc<len(interacted_times) and qtime >= interacted_times[qtime_loc]:
        qtime_loc+=1

    if big_or_small_or_history == 'big':
        if recall_type == 'i2iw10' or recall_type== 'i2iw02':
            recall_left_max_road_num = 15
            recall_right_max_road_num = 15
            recall_left_max_num_each_road = 250
            recall_right_max_num_each_road = 250
        elif recall_type == 'b2b':
            recall_left_max_road_num = 15
            recall_right_max_road_num = 15
            recall_left_max_num_each_road = 350
            recall_right_max_num_each_road = 350
        elif recall_type == 'i2i2i_new':
            recall_left_max_road_num = 15
            recall_right_max_road_num = 15
            recall_left_max_num_each_road = 150
            recall_right_max_num_each_road = 150
        else:
            print(1/0)
    elif big_or_small_or_history == 'small':
        if recall_type == 'i2iw10' or recall_type== 'i2iw02':
            recall_left_max_road_num = 10
            recall_right_max_road_num = 10
            recall_left_max_num_each_road = 100
            recall_right_max_num_each_road = 100
        elif recall_type == 'b2b':
            recall_left_max_road_num = 10
            recall_right_max_road_num = 10
            recall_left_max_num_each_road = 100
            recall_right_max_num_each_road = 100
        elif recall_type == 'i2i2i_new':
            recall_left_max_road_num = 10
            recall_right_max_road_num = 10
            recall_left_max_num_each_road = 100
            recall_right_max_num_each_road = 100
        else:
            print(1/0)
    elif big_or_small_or_history == 'history':
        if recall_type == 'i2iw10' or recall_type== 'i2iw02':
            recall_left_max_road_num = 25
            recall_right_max_road_num = 0
            recall_left_max_num_each_road = 250
            recall_right_max_num_each_road = 0
        elif recall_type == 'b2b':
            recall_left_max_road_num = 25
            recall_right_max_road_num = 0
            recall_left_max_num_each_road = 350
            recall_right_max_num_each_road = 0
        elif recall_type == 'i2i2i_new':
            recall_left_max_road_num = 25
            recall_right_max_road_num = 0
            recall_left_max_num_each_road = 150
            recall_right_max_num_each_road = 0
        else:
            print(1/0)
    elif big_or_small_or_history == 'his30-fut0':
        if recall_type == 'i2iw10' or recall_type== 'i2iw02':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 0
            recall_left_max_num_each_road = 250
            recall_right_max_num_each_road = 250
        elif recall_type == 'b2b':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 0
            recall_left_max_num_each_road = 350
            recall_right_max_num_each_road = 350
        elif recall_type == 'i2i2i_new':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 0
            recall_left_max_num_each_road = 150
            recall_right_max_num_each_road = 150
        else:
            print(1/0)
    elif big_or_small_or_history == 'his30-fut1':
        if recall_type == 'i2iw10' or recall_type== 'i2iw02':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 1
            recall_left_max_num_each_road = 250
            recall_right_max_num_each_road = 250
        elif recall_type == 'b2b':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 1
            recall_left_max_num_each_road = 350
            recall_right_max_num_each_road = 350
        elif recall_type == 'i2i2i_new':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 1
            recall_left_max_num_each_road = 150
            recall_right_max_num_each_road = 150
        else:
            print(1/0)
    elif big_or_small_or_history == 'his30-fut4':
        if recall_type == 'i2iw10' or recall_type== 'i2iw02':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 4
            recall_left_max_num_each_road = 250
            recall_right_max_num_each_road = 250
        elif recall_type == 'b2b':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 4
            recall_left_max_num_each_road = 350
            recall_right_max_num_each_road = 350
        elif recall_type == 'i2i2i_new':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 4
            recall_left_max_num_each_road = 150
            recall_right_max_num_each_road = 150
        else:
            print(1/0)
    elif big_or_small_or_history == 'his30-fut8':
        if recall_type == 'i2iw10' or recall_type== 'i2iw02':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 8
            recall_left_max_num_each_road = 250
            recall_right_max_num_each_road = 250
        elif recall_type == 'b2b':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 8
            recall_left_max_num_each_road = 350
            recall_right_max_num_each_road = 350
        elif recall_type == 'i2i2i_new':
            recall_left_max_road_num = 30
            recall_right_max_road_num = 8
            recall_left_max_num_each_road = 150
            recall_right_max_num_each_road = 150
        else:
            print(1/0)
    else:
        print(1/0)

    
    l_cans_loc = []
    r_cans_loc = []
    num = 1
    while num <= recall_left_max_road_num :
        now_loc = qtime_loc-num
        if now_loc < 0:
            break
        l_cans_loc.append( now_loc )
        num += 1
    num = 0
    while num <= recall_right_max_road_num-1:
        now_loc = qtime_loc+num
        if now_loc >= len(interacted_items):
            break
        r_cans_loc.append( now_loc )
        num += 1
    
    if get_sum:
        multi_road_result = {}
    else:
        multi_road_result = []
    for i in range(len(l_cans_loc)):
        item = interacted_items[ l_cans_loc[i] ]
        time = interacted_times[ l_cans_loc[i] ]
        each_road_result = []
        
        loc_weight = 0.7**i
        if loc_weight<=0.1:
            loc_weight = 0.1
        time_weight = (1 - abs( qtime - time ) * 100)
        if time_weight<=0.1:
            time_weight = 0.1
        
        if item not in sim_item_corr:
            continue
        for j, wij in sim_item_corr[ item ].items(): 
            sim_weight = wij
            rank_weight = sim_weight * loc_weight * time_weight
            each_road_result.append( ( j, sim_weight, loc_weight, time_weight, rank_weight, item,
                                       l_cans_loc[i], time, qtime_loc, qtime ) )
        each_road_result.sort(key=lambda x:x[1], reverse=True)
        each_road_result = each_road_result[0:recall_left_max_num_each_road]
        
        if get_sum:
            for idx,k in enumerate(each_road_result):
                if k[0] not in multi_road_result:
                    multi_road_result[k[0]] = k[1:]
                else:
                    t1 = multi_road_result[k[0]]
                    t2 = k[1:]
                    multi_road_result[k[0]] = ( t1[0]+t2[0] , t1[1], t1[2], t1[3]+t2[3], t1[4],
                                                t1[5], t1[6], t1[7], t1[8] )
        else:
            multi_road_result += each_road_result

    for i in range(len(r_cans_loc)):
        item = interacted_items[ r_cans_loc[i] ]
        time = interacted_times[ r_cans_loc[i] ]
        each_road_result = []
        
        loc_weight = 0.7**(i)
        if loc_weight<=0.1:
            loc_weight = 0.1
        time_weight = (1 - abs( qtime - time ) * 100)
        if time_weight<=0.1:
            time_weight = 0.1
        
        if item not in sim_item_corr:
            continue
        for j, wij in sim_item_corr[ item ].items(): 
            sim_weight = wij
            rank_weight = sim_weight * loc_weight * time_weight
            each_road_result.append( ( j, sim_weight, loc_weight, time_weight, rank_weight, item,
                                       r_cans_loc[i], time, qtime_loc, qtime ) )
        each_road_result.sort(key=lambda x:x[1], reverse=True)
        each_road_result = each_road_result[0:recall_right_max_num_each_road]
        
        if get_sum:
            for idx,k in enumerate(each_road_result):
                if k[0] not in multi_road_result:
                    multi_road_result[k[0]] = k[1:]
                else:
                    t1 = multi_road_result[k[0]]
                    t2 = k[1:]
                    multi_road_result[k[0]] = ( t1[0]+t2[0] , t1[1], t1[2], t1[3]+t2[3], t1[4],
                                                t1[5], t1[6], t1[7], t1[8] )
        else:
            multi_road_result += each_road_result
    if get_sum:
        multi_road_result_t = sorted(multi_road_result.items(), key=lambda i: i[1][3], reverse=True)
        multi_road_result = []
        for q in multi_road_result_t:
            multi_road_result.append( (q[0],)+q[1] )
    else:
        multi_road_result.sort(key=lambda x:x[4], reverse=True)

    return multi_road_result

def i2i_w02_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    all_pair_num = 0
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                
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
                    
    for i, related_items in sim_item.items():  
        for j, cij in related_items.items():  
            sim_item[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)
    
    print('all_pair_num',all_pair_num)
    for key in sim_item.keys():
        t = sim_item[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:i2i_sim_limit]:
            res[i[0]]=i[1]
        sim_item[key] = res
        
    user2recall = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall[(user,qtime)] = recommend(sim_item,user_item_dict,user_time_dict,user,qtime,0.7,'i2iw02')
        if len(user2recall) % 100 ==0:
            print(len(user2recall))
    
    phase_ndcg_pred_answer = []
    answers_source = []
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    for predict_stage in range(cur_stage+1):
        predictions = []
        pos = []
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict
        print(f'i2i_w02 recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            recall_items = user2recall[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:
                        new_recall.append(re)
            
            predictions.append(new_recall)
            pos.append(it)
            if len(predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(predictions)}, tot: {tot}')
        
        item_degree = phase_item_degree[predict_stage]
        if mode == 'test':
            answers = [ (p, np.nan) for p in pos ]
        else:
            answers = [(p, item_degree[p]) for p in pos]
        
        phase_ndcg_pred_answer.append( predictions )
        answers_source.append( answers )
    
    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i_w02_recall_scoure_path.format( mode, cur_stage,'sum' ) )    
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i_w02_recall_scoure_path.format( mode, cur_stage,'nosum' ) )    
    utils.dump_pickle( answers_source, answer_source_path.format( mode, cur_stage ) )    

def b2b_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    blend_sim = utils.load_sim(item_blend_sim_path)
    blend_score = {}
    
    for item in blend_sim:
        i = item[0]
        blend_score.setdefault(i,{})
        for j,cij in item[1][:b2b_sim_limit]:
            blend_score[i][j] = cij
    
    user2recall_blendsim = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall_blendsim[(user,qtime)] = recommend(blend_score,user_item_dict,user_time_dict,user,qtime,0.7,'b2b')
        if len(user2recall_blendsim) % 100 ==0:
            print(len(user2recall_blendsim))
    
    phase_ndcg_pred_answer = []

    #phase_ndcg_pred_answer -> [ 0,1,2,3 .. 9]
    #phase_ndcg_pred_answer[0] -> [ 1,2,3,4,5,6,7... ]
    #1.7w phase_ndcg_pred_answer[0][0]->[ (2,1.2,3.4), (3,4,3) ] -> 1.6k
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    for predict_stage in range(cur_stage+1):
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict
        
        blend_predictions = []
        print(f'b2b recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            recall_items = user2recall_blendsim[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:# and re in feat_item_set:
                        new_recall.append(re)
                
            blend_predictions.append(new_recall)
            if len(blend_predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(blend_predictions)}, tot: {tot}')
                
        phase_ndcg_pred_answer.append( blend_predictions )

    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, b2b_recall_scoure_path.format( mode, cur_stage, 'sum' ) )    
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, b2b_recall_scoure_path.format( mode, cur_stage, 'nosum' ) )      
    
def i2i2i_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    all_pair_num = 0
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                
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
                    
    for i, related_items in sim_item.items():  
        for j, cij in related_items.items():  
            sim_item[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)
    
    print('all_pair_num',all_pair_num)
    for key in sim_item.keys():
        t = sim_item[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:i2i2i_sim_limit]:
            res[i[0]]=i[1]
        sim_item[key] = res
            
    import time
    t1 = time.time()
    sim_item_p2 = {}
    for idx,item1 in enumerate( sim_item.keys() ):
        if idx%10000==0:
            t2 = time.time()
            print( f'use time {t2-t1} for 10000, now {idx} , tot {len(sim_item.keys())}' )
            t1 = t2
        sim_item_p2.setdefault(item1, {})
        for item2 in sim_item[item1].keys():
            if item2 == item1:
                continue
            for item3 in sim_item[item2].keys():
                if item3 == item1 or item3 == item2:
                    continue
                sim_item_p2[item1].setdefault(item3, 0)
                sim_item_p2[item1][item3] += sim_item[item1][item2]*sim_item[item2][item3]

    user2recall_i2i2i = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall_i2i2i[(user,qtime)] = recommend(sim_item_p2,user_item_dict,user_time_dict,user,qtime,0.7,'i2i2i')
        if len(user2recall_i2i2i) % 100 ==0:
            print(len(user2recall_i2i2i))        
    
    phase_ndcg_pred_answer = []
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    for predict_stage in range(cur_stage+1):
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict

        i2i2i_predictions = []
        print(f'i2i2i recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            
            recall_items = user2recall_i2i2i[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:# and re in feat_item_set:
                        new_recall.append(re)
                
            i2i2i_predictions.append(new_recall)
            
            if len(i2i2i_predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(i2i2i_predictions)}, tot: {tot}')
                
        phase_ndcg_pred_answer.append( i2i2i_predictions )#,text_predictions,image_predictions) )
    
    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i2i_recall_scoure_path.format( mode, cur_stage, 'sum' ) ) 
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i2i_recall_scoure_path.format( mode, cur_stage, 'nosum' ) ) 

def i2i2i_new_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    all_pair_num = 0
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                
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
                    
    for i, related_items in sim_item.items():  
        for j, cij in related_items.items():  
            sim_item[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)
    
    
    i2i_sim = {}
    print('all_pair_num',all_pair_num)
    for key in sim_item.keys():
        t = sim_item[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = set()
        for i in t[0:i2i_sim_limit]:
            res.add( i[0] )
        i2i_sim[key] = res

        res = {}
        for i in t[0:i2i2i_new_sim_limit]:
            res[i[0]]=i[1]
        sim_item[key] = res
            
    import time
    t1 = time.time()
    sim_item_p2 = {}
    for idx,item1 in enumerate( sim_item.keys() ):
        if idx%10000==0:
            t2 = time.time()
            print( f'use time {t2-t1} for 10000, now {idx} , tot {len(sim_item.keys())}' )
            t1 = t2
        sim_item_p2.setdefault(item1, {})
        for item2 in sim_item[item1].keys():
            if item2 == item1:
                continue
            for item3 in sim_item[item2].keys():
                if item3 == item1 or item3 == item2:
                    continue
                if item3 in i2i_sim[item1]:
                    continue
                sim_item_p2[item1].setdefault(item3, 0)
                sim_item_p2[item1][item3] += sim_item[item1][item2]*sim_item[item2][item3]

    user2recall_i2i2i = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall_i2i2i[(user,qtime)] = recommend(sim_item_p2,user_item_dict,user_time_dict,user,qtime,0.7,'i2i2i_new')
        if len(user2recall_i2i2i) % 100 ==0:
            print(len(user2recall_i2i2i))        
    
    phase_ndcg_pred_answer = []
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    for predict_stage in range(cur_stage+1):
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict

        i2i2i_predictions = []
        print(f'i2i2i recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            
            recall_items = user2recall_i2i2i[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:# and re in feat_item_set:
                        new_recall.append(re)
                
            i2i2i_predictions.append(new_recall)
            
            if len(i2i2i_predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(i2i2i_predictions)}, tot: {tot}')
                
        phase_ndcg_pred_answer.append( i2i2i_predictions )#,text_predictions,image_predictions) )
    
    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i2i_new_recall_scoure_path.format( mode, cur_stage, 'sum' ) ) 
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i2i_new_recall_scoure_path.format( mode, cur_stage, 'nosum' ) ) 

def i2i_w10_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    all_pair_num = 0
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                
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
                    
    for i, related_items in sim_item.items():  
        for j, cij in related_items.items():  
            sim_item[i][j] = cij / (item_cnt[i] * item_cnt[j])
    
    print('all_pair_num',all_pair_num)
    for key in sim_item.keys():
        t = sim_item[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:i2i_sim_limit]:
            res[i[0]]=i[1]
        sim_item[key] = res
        
    user2recall = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall[(user,qtime)] = recommend(sim_item,user_item_dict,user_time_dict,user,qtime,0.7,'i2iw10')
        if len(user2recall) % 100 ==0:
            print(len(user2recall))
    
    phase_ndcg_pred_answer = []
    answers_source = []
    for predict_stage in range(cur_stage+1):
        predictions = []
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict
        print(f'i2i_w10 recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            recall_items = user2recall[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:
                        new_recall.append(re)
            
            predictions.append(new_recall)
            if len(predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(predictions)}, tot: {tot}')
    
        phase_ndcg_pred_answer.append( predictions )
    
    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i_w10_recall_scoure_path.format( mode, cur_stage, 'sum' ) )   
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i_w10_recall_scoure_path.format( mode, cur_stage, 'nosum' ) )    

def b2b2b_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    blend_sim = utils.load_sim(item_blend_sim_path)
    blend_score = {}
    
    for item in blend_sim:
        i = item[0]
        blend_score.setdefault(i,{})
        for j,cij in item[1][:b2b2b_sim_limit]:
            blend_score[i][j] = cij

    import time
    t1 = time.time()
    blend_score_2 = {}
    for idx,item1 in enumerate( blend_score.keys() ):
        if idx%10000==0:
            t2 = time.time()
            print( f'use time {t2-t1} for 10000, now {idx} , tot {len(blend_score.keys())}' )
            t1 = t2
        blend_score_2.setdefault(item1, {})
        for item2 in blend_score[item1].keys():
            if item2 == item1:
                continue
            for item3 in blend_score[item2].keys():
                if item3 == item1 or item3 == item2:
                    continue
                blend_score_2[item1].setdefault(item3, 0)
                blend_score_2[item1][item3] += blend_score[item1][item2]*blend_score[item2][item3]
    
    user2recall_blendsim = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall_blendsim[(user,qtime)] = recommend(blend_score_2,user_item_dict,user_time_dict,user,qtime,0.7,'b2b2b')
        if len(user2recall_blendsim) % 100 ==0:
            print(len(user2recall_blendsim))
       
    phase_ndcg_pred_answer = []
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    for predict_stage in range(cur_stage+1):
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict
        
        blend_predictions = []
        print(f'b2b2b recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            recall_items = user2recall_blendsim[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:# and re in feat_item_set:
                        new_recall.append(re)
                
            blend_predictions.append(new_recall)
            if len(blend_predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(blend_predictions)}, tot: {tot}')
                
        phase_ndcg_pred_answer.append( blend_predictions )
    
    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, b2b2b_recall_scoure_path.format( mode, cur_stage, 'sum' ) )  
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, b2b2b_recall_scoure_path.format( mode, cur_stage, 'nosum' ) )  

def i2i2b_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    all_pair_num = 0
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                
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
                    
    for i, related_items in sim_item.items():  
        for j, cij in related_items.items():  
            sim_item[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)
    
    print('all_pair_num',all_pair_num)
    for key in sim_item.keys():
        t = sim_item[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:i2i2b_i_sim_limit]:
            res[i[0]]=i[1]
        sim_item[key] = res

    blend_sim = utils.load_sim(item_blend_sim_path)
    blend_score = {}
    
    for item in blend_sim:
        i = item[0]
        blend_score.setdefault(i,{})
        for j,cij in item[1][:i2i2b_b_sim_limit]:
            blend_score[i][j] = cij

    import time
    t1 = time.time()
    blend_score_2 = {}
    for idx,item1 in enumerate( sim_item.keys() ):
        if idx%10000==0:
            t2 = time.time()
            print( f'use time {t2-t1} for 10000, now {idx} , tot {len(sim_item.keys())}' )
            t1 = t2
        blend_score_2.setdefault(item1, {})
        for item2 in sim_item[item1].keys():
            if item2 == item1:
                continue
            if item2 in blend_score.keys():
                for item3 in blend_score[item2].keys():
                    if item3 == item1 or item3 == item2:
                        continue
                    blend_score_2[item1].setdefault(item3, 0)
                    blend_score_2[item1][item3] += sim_item[item1][item2]*blend_score[item2][item3]
    
    user2recall_blendsim = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall_blendsim[(user,qtime)] = recommend(blend_score_2,user_item_dict,user_time_dict,user,qtime,0.7,'i2i2b')
        if len(user2recall_blendsim) % 100 ==0:
            print(len(user2recall_blendsim))
       
    phase_ndcg_pred_answer = []
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    for predict_stage in range(cur_stage+1):
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict
        
        blend_predictions = []
        print(f'i2i2b recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            recall_items = user2recall_blendsim[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:# and re in feat_item_set:
                        new_recall.append(re)
                
            blend_predictions.append(new_recall)
            if len(blend_predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(blend_predictions)}, tot: {tot}')
                
        phase_ndcg_pred_answer.append( blend_predictions )
    
    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i2b_recall_scoure_path.format( mode, cur_stage, 'sum' ) )  
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, i2i2b_recall_scoure_path.format( mode, cur_stage, 'nosum' ) )  

def b2b2i_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    
    all_pair_num = 0
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1
                
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                
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
                    
    for i, related_items in sim_item.items():  
        for j, cij in related_items.items():  
            sim_item[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)
    
    print('all_pair_num',all_pair_num)
    for key in sim_item.keys():
        t = sim_item[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:b2b2i_i_sim_limit]:
            res[i[0]]=i[1]
        sim_item[key] = res

    blend_sim = utils.load_sim(item_blend_sim_path)
    blend_score = {}
    
    for item in blend_sim:
        i = item[0]
        blend_score.setdefault(i,{})
        for j,cij in item[1][:b2b2i_b_sim_limit]:
            blend_score[i][j] = cij

    import time
    t1 = time.time()
    blend_score_2 = {}
    for idx,item1 in enumerate( blend_score.keys() ):
        if idx%10000==0:
            t2 = time.time()
            print( f'use time {t2-t1} for 10000, now {idx} , tot {len(blend_score.keys())}' )
            t1 = t2
        blend_score_2.setdefault(item1, {})
        for item2 in blend_score[item1].keys():
            if item2 == item1:
                continue
            if item2 in sim_item.keys():
                for item3 in sim_item[item2].keys():
                    if item3 == item1 or item3 == item2:
                        continue
                    blend_score_2[item1].setdefault(item3, 0)
                    blend_score_2[item1][item3] += blend_score[item1][item2]*sim_item[item2][item3]
    
    user2recall_blendsim = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall_blendsim[(user,qtime)] = recommend(blend_score_2,user_item_dict,user_time_dict,user,qtime,0.7,'b2b2i')
        if len(user2recall_blendsim) % 100 ==0:
            print(len(user2recall_blendsim))
    
    phase_ndcg_pred_answer = []
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    for predict_stage in range(cur_stage+1):
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict
        
        blend_predictions = []
        print(f'b2b2i recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            recall_items = user2recall_blendsim[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:# and re in feat_item_set:
                        new_recall.append(re)
                
            blend_predictions.append(new_recall)
            if len(blend_predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(blend_predictions)}, tot: {tot}')
                
        phase_ndcg_pred_answer.append( blend_predictions )
    
    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, b2b2i_recall_scoure_path.format( mode, cur_stage, 'sum' ) )  
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, b2b2i_recall_scoure_path.format( mode, cur_stage, 'nosum' ) )  

def b2bl2_recall(df_train, df_train_stage, df, df_stage):
    user_item_ = df_train.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_time_ = df_train.groupby('user_id')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user_id'], user_time_['time']))
    blend_sim = utils.load_sim(item_text_l2_sim_path)

    blend_score = {}
    
    for item in blend_sim:
        i = item[0]
        blend_score.setdefault(i,{})
        for j,cij in item[1][:b2b_sim_limit]:
            blend_score[i][j] = cij
    
    user2recall_blendsim = {}
    for user,qtime in zip(df['user_id'],df['time']):
        user2recall_blendsim[(user,qtime)] = recommend(blend_score,user_item_dict,user_time_dict,user,qtime,0.7,'b2bl2')
        if len(user2recall_blendsim) % 100 ==0:
            print(len(user2recall_blendsim))
    
    phase_ndcg_pred_answer = []
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    for predict_stage in range(cur_stage+1):
        df_now = df_stage[df_stage['stage'] == predict_stage]
        df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
        stage_items = set(df_train['item_id'])
        cur_user_item_dict = user_item_dict
        
        blend_predictions = []
        print(f'b2b recall start {predict_stage}')
        for user_id,it,qtime in zip(df_now['user_id'],df_now['item_id'],df_now['time']):
            recall_items = user2recall_blendsim[(user_id,qtime)]
            new_recall = []
            for re in recall_items:
                if re[0] == it:
                    new_recall.append(re)
                elif (user_id not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user_id]):
                    if re[0] in stage_items:# and re in feat_item_set:
                        new_recall.append(re)
                
            blend_predictions.append(new_recall)
            if len(blend_predictions)%1000 == 0: 
                tot = len(df_now['user_id'])
                print(f'now: {len(blend_predictions)}, tot: {tot}')
                
        phase_ndcg_pred_answer.append( blend_predictions )

    if get_sum:
        utils.dump_pickle( phase_ndcg_pred_answer, b2bl2_recall_scoure_path.format( mode, cur_stage, 'sum' ) )    
    else:
        utils.dump_pickle( phase_ndcg_pred_answer, b2bl2_recall_scoure_path.format( mode, cur_stage, 'nosum' ) )   

def calculate_score_for_each():
    recall_nums = [50,1000]
    answer_source = utils.load_pickle( answer_source_path.format( 'valid', cur_stage, 'sum' ) )
    i2i_w02_recall_source = utils.load_pickle( i2i_w02_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    b2b_recall_source = utils.load_pickle( b2b_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    i2i2i_recall_source = utils.load_pickle( i2i2i_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    i2i_w10_recall_source = utils.load_pickle( i2i_w10_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    b2b2b_recall_source = utils.load_pickle( b2b2b_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    i2i2b_recall_source = utils.load_pickle( i2i2b_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    b2b2i_recall_source = utils.load_pickle( b2b2i_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )


    recall_sources = [ i2i_w02_recall_source, b2b_recall_source, i2i2i_recall_source,
                       i2i_w10_recall_source, b2b2b_recall_source, i2i2b_recall_source,
                       b2b2i_recall_source ]
    recall_source_names = [ 'i2i_w02_recall_source', 'b2b_recall_source', 'i2i2i_recall_source',
                            'i2i_w10_recall_source', 'b2b2b_recall_source', 'i2i2b_recall_source',
                            'b2b2i_recall_source' ]

    result = {}
    for recall_source_name, recall_source in zip( recall_source_names, recall_sources ):
        result[recall_source_name] = {}
        for recall_num in recall_nums:
            all_scores = []
            recall = []
            #num1 = int(recall_num/5*3)
            #num2 = recall_num - num1
            num1 = recall_num
            num2 = recall_num
            num3 = recall_num
            for sta in range(cur_stage+1):
                answers = answer_source[sta]
                predictions = recall_source[sta]
                weight = [num1,num2,num3]
                new_predictions = []
                re_new_predictions = []
                #for pred,text_pred,image_pred in zip(predictions,text_predictions,image_predictions):
            
                for pred in predictions:
                    new_pred = []   
                    for j in pred:
                        if len(new_pred) < weight[0]:
                            flag = 0
                            for k in new_pred:
                                if j[0] == k[0]:
                                    flag = 1
                                    break
                            if flag==0:
                                new_pred.append( j )
                    '''
                    for j in blend_pred:
                        if len(new_pred) < weight[0]+weight[1]:
                            flag = 0
                            for k in new_pred:
                                if j[0] == k[0]:
                                    flag = 1
                                    break
                            if flag==0:
                                new_pred.append( j )
                    
                    
                    for j in i2i2i_pred:
                        if len(new_pred) < weight[0]+weight[1]+weight[2]:
                            flag = 0
                            for k in new_pred:
                                if j[0] == k[0]:
                                    flag = 1
                                    break
                            if flag==0:
                                new_pred.append( j )
                    '''
                    s_new_pred = [ ve[0] for ve in new_pred ]
                    new_predictions.append(s_new_pred+[0]*(recall_num-len(new_pred)))
                    re_new_predictions.append( new_pred+[(0,0)]*(recall_num-len(new_pred)) )
                
                recall.append( (answers, re_new_predictions) )
                scores = ndcg_tools.evaluate_each_phase(new_predictions, answers, at=recall_num)
                print(scores)
                all_scores.append(scores)
            result[recall_source_name][recall_num] = (all_scores, np.array(all_scores).sum(axis=0))
            print(f'{recall_source_name} - {recall_num} all_scores_sum',np.array(all_scores).sum(axis=0))
    return result

def calculate_score_for_merge():
    recall_nums = [50,1000]

    if get_sum:
        sum_mode = 'sum'
    else:
        sum_mode = 'nosum'

    answer_source = utils.load_pickle( answer_source_path.format( 'valid', cur_stage, sum_mode ) )
    i2i_w02_recall_source = utils.load_pickle( i2i_w02_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    b2b_recall_source = utils.load_pickle( b2b_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    i2i2i_recall_source = utils.load_pickle( i2i2i_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    i2i_w10_recall_source = utils.load_pickle( i2i_w10_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    b2b2b_recall_source = utils.load_pickle( b2b2b_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    i2i2b_recall_source = utils.load_pickle( i2i2b_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    b2b2i_recall_source = utils.load_pickle( b2b2i_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )

    recall_sources = [ i2i_w02_recall_source, b2b_recall_source, i2i2i_recall_source,
                        i2i_w10_recall_source ]
                     #  i2i_w10_recall_source, b2b2b_recall_source, i2i2b_recall_source,
                     #  b2b2i_recall_source ]
    recall_source_names = [ 'i2i_w02_recall_source', 'b2b_recall_source', 'i2i2i_recall_source',
                            'i2i_w10_recall_source' ]
                     #       'i2i_w10_recall_source', 'b2b2b_recall_source', 'i2i2b_recall_source',
                     #       'b2b2i_recall_source' ]

    for recall_num in recall_nums:
        all_scores = []
        num1 = recall_num
        num2 = recall_num
        num3 = recall_num
        num4 = recall_num
        for sta in range(cur_stage+1):
            answers = answer_source[sta]

            weight = [num1,num2,num3,num4]
            new_predictions = []
            re_new_predictions = []
    
            i2i = i2i_w02_recall_source[sta]
            b2b = b2b_recall_source[sta]
            i2i2i = i2i2i_recall_source[sta]
            i2i_w10 = i2i_w10_recall_source[sta]
            for pred1,pred2,pred3,pred4 in zip( i2i,b2b,i2i2i,i2i_w10 )  :
                new_pred = []   
                for j in pred1:
                    if len(new_pred) < weight[0]:
                        flag = 0
                        for k in new_pred:
                            if j[0] == k[0]:
                                flag = 1
                                break
                        if flag==0:
                            new_pred.append( j )

                for j in pred2:
                    if len(new_pred) < weight[0]+weight[1]:
                        flag = 0
                        for k in new_pred:
                            if j[0] == k[0]:
                                flag = 1
                                break
                        if flag==0:
                            new_pred.append( j )
                
                for j in pred3:
                    if len(new_pred) < weight[0]+weight[1]+weight[2]:
                        flag = 0
                        for k in new_pred:
                            if j[0] == k[0]:
                                flag = 1
                                break
                        if flag==0:
                            new_pred.append( j )
                
                for j in pred4:
                    if len(new_pred) < weight[0]+weight[1]+weight[2]+weight[3]:
                        flag = 0
                        for k in new_pred:
                            if j[0] == k[0]:
                                flag = 1
                                break
                        if flag==0:
                            new_pred.append( j )
                s_new_pred = [ ve[0] for ve in new_pred ]
                new_predictions.append(s_new_pred+[0]*(len(recall_sources)*recall_num-len(new_pred)))

            scores = ndcg_tools.evaluate_each_phase(new_predictions, answers, at=len(recall_sources)*recall_num)
            print(scores)
            all_scores.append(scores)
        print(f'{recall_source_names} - {len(recall_sources)*recall_num} all_scores_sum',np.array(all_scores).sum(axis=0))

def calculate_hitrate_for_merge():
    if get_sum:
        sum_mode = 'sum'
    else:
        sum_mode = 'nosum'

    answer_source = utils.load_pickle( answer_source_path.format( 'valid', cur_stage, sum_mode ) )
    i2i_w02_recall_source = utils.load_pickle( i2i_w02_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    b2b_recall_source = utils.load_pickle( b2b_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    i2i_w10_recall_source = utils.load_pickle( i2i_w10_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    i2i2i_new_recall_source = utils.load_pickle( i2i2i_new_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) ) 
    #i2i2i_recall_source = utils.load_pickle( i2i2i_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    #i2i2b_recall_source = utils.load_pickle( i2i2b_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    #b2b2i_recall_source = utils.load_pickle( b2b2i_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    #b2b2b_recall_source = utils.load_pickle( b2b2b_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    #b2bl2_recall_source = utils.load_pickle( b2bl2_recall_scoure_path.format( 'valid', cur_stage, sum_mode ) )
    #import pdb
    #pdb.set_trace()
    #filter 
    '''
    sources = [ i2i_w02_recall_source, b2b_recall_source, i2i2i_new_recall_source ]
    for source in sources:
        for sta in range(cur_stage+1):
            this_source = source[sta]            
            for idx in range(len(this_source)):
                new = []
                from collections import defaultdict
                diff_cnt = defaultdict(int)
                for p in this_source[idx]:

                    if p[6] < p[8]:
                        loc_diff = p[8]-p[6]
                    else:
                        loc_diff = p[6]-p[8]+1
                    if loc_diff<=20:
                        if diff_cnt[p[8]-p[6]]<=num:
                            new.append( p )
                            diff_cnt[p[8]-p[6]] += 1
                        
                this_source[idx] = new
    '''

    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage))
    df_valid = utils.load_pickle(all_valid_stage_data_path.format(cur_stage))
    phase2valid_item_degree = {}
    phase2median = {}

    for sta in range(cur_stage+1):
        cur_df_valid = df_valid[df_valid['stage']==sta]
        
        items = cur_df_valid['item_id'].values
        item_degree = phase_item_degree[sta]
        
        list_item_degress = []
        for item_id in items:
            list_item_degress.append(item_degree[item_id])
            
        list_item_degress.sort()
        median_item_degree = list_item_degress[len(list_item_degress) // 2]
        phase2median[sta] = median_item_degree
        for item in items:
            phase2valid_item_degree[(sta,item)] = item_degree[item]


    hit_tot = 0
    tot = 0
    rare_hit_tot = 0
    rare_tot = 0
    data_num = [0]*3
    for sta in range(cur_stage+1):
        answers = answer_source[sta]
        tot += len(answers)
        #i2i = i2i_w02_recall_source[sta]
        i2i = i2i_w10_recall_source[sta]
        b2b = b2b_recall_source[sta]
        
        #i2i2i = i2i2i_recall_source[sta]
        #i2i_w10 = i2i_w10_recall_source[sta]
        #i2i2b = i2i2b_recall_source[sta]
        #b2b2b = b2b2b_recall_source[sta]
        #b2b2i = b2b2i_recall_source[sta]
        #b2bl2 = b2bl2_recall_source[sta]
        i2i2i_new = i2i2i_new_recall_source[sta]
        for ans1,pred1,pred2,pred3 in zip( answers,i2i,b2b,i2i2i_new ): #i2i_w10,i2i2b,b2b2b,b2b2i,b2bl2 )  :
            item = ans1[0]
            if phase2valid_item_degree[(sta,item)]<=phase2median[sta]:
                rare_tot += 1
            flag = 0
            pred_list = [pred1,pred2,pred3]
            for idx,pred in enumerate(pred_list):
                data_num[idx] += len(pred)
            for pred in pred_list:
                for j in pred:
                    if item==j[0]:
                        hit_tot+=1
                        if phase2valid_item_degree[(sta,item)]<=phase2median[sta]:
                            rare_hit_tot += 1
                        flag = 1
                        break
                if flag:
                    break
            if flag:
                continue
            
    print('full hit: ', hit_tot/tot)
    print('rare hit: ', rare_hit_tot/rare_tot)
    print('data_num: ', data_num)
    print('sum data_num: ', np.sum(data_num))
def calculate_x():
    recall_nums = [50,1000]
    answer_source = utils.load_pickle( answer_source_path.format( 'valid', cur_stage, 'sum' ) )
    i2i_w02_recall_source = utils.load_pickle( i2i_w02_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    b2b_recall_source = utils.load_pickle( b2b_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    i2i2i_recall_source = utils.load_pickle( i2i2i_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    i2i_w10_recall_source = utils.load_pickle( i2i_w10_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    b2b2b_recall_source = utils.load_pickle( b2b2b_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    i2i2b_recall_source = utils.load_pickle( i2i2b_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )
    b2b2i_recall_source = utils.load_pickle( b2b2i_recall_scoure_path.format( 'valid', cur_stage, 'sum' ) )

    result = {}
    recall_sources = [ i2i_w02_recall_source, b2b_recall_source, i2i2i_recall_source,
                       i2i_w10_recall_source, b2b2b_recall_source, i2i2b_recall_source,
                       b2b2i_recall_source ]
    recall_source_names = [ 'i2i_w02_recall_source', 'b2b_recall_source', 'i2i2i_recall_source',
                            'i2i_w10_recall_source', 'b2b2b_recall_source', 'i2i2b_recall_source',
                            'b2b2i_recall_source' ]

    
    result = { 'recall_num':{}, 'recall_x':{} }
    for recall_source_name_1, recall_source_1 in zip( recall_source_names, recall_sources ):
        result['recall_num'][recall_source_name_1] = 0
        result['recall_x'][recall_source_name_1] = {}
        
        for sta in range( cur_stage+1 ):
            for user in range( len(recall_source_1[sta]) ):
                item1s = []
                for j in range( len(recall_source_1[sta][user]) ):
                    item1s.append( recall_source_1[sta][user][j][0] )
                if len(item1s) != len(set(item1s)):
                    print('mismatch')
                    print(1/0)
                result['recall_num'][recall_source_name_1] += len( set(item1s) )

        for recall_source_name_2, recall_source_2 in zip( recall_source_names, recall_sources ):
            result['recall_x'][recall_source_name_1][recall_source_name_2] = 0  
            ans = []
            for sta in range( cur_stage+1 ):
                for user in range( len(recall_source_1[sta]) ):    
                    item1s = []
                    item2s = []
                    for j in range( len(recall_source_1[sta][user]) ):
                        item1s.append( recall_source_1[sta][user][j][0] )
                    for j in range( len(recall_source_2[sta][user]) ):
                        item2s.append( recall_source_2[sta][user][j][0] )
                    if len( set(item1s) ) != len(item1s):
                        print('mismatch')
                        print(1/0)
                    if (len( set(item1s)|set(item2s) )) != 0:
                        ans.append(  (len( set(item1s)&set(item2s) )) /  (len( set(item1s)|set(item2s) )) )
                    else:
                        ans.append(1.0)
            result['recall_x'][recall_source_name_1][recall_source_name_2] = np.mean(ans)             
    
    print('--------------------recall_num-----------------')
    for key1 in result['recall_num'].keys():
        print( f'{key1}: ', result['recall_num'][key1] )

    print('--------------------recall_x------------------')
    for key1 in result['recall_x'].keys():
        for key2 in result['recall_x'][key1].keys():
            print( f'{key1}-{key2}: ', result['recall_x'][key1][key2] )
    return result


if get_sum:
    print('now run sum_mode: sum')
else:
    print('now run sum_mode: nosum')

recall_diff_road_list = ['i2i_w02','i2i_w10','b2b','i2i2i_new']
if mode == 'valid':
    df_train = utils.load_pickle(all_train_data_path.format(cur_stage))
    df_train_stage = utils.load_pickle(all_train_stage_data_path.format(cur_stage))
    df = utils.load_pickle(all_valid_data_path.format(cur_stage))
    df_stage = utils.load_pickle(all_valid_stage_data_path.format(cur_stage))
else:
    df_train = utils.load_pickle(online_all_train_data_path.format(cur_stage))
    df_train_stage = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    df = utils.load_pickle(online_all_test_data_path.format(cur_stage))
    df['item_id'] = np.nan
    df_stage = utils.load_pickle(online_all_test_data_path.format(cur_stage))
    df_stage['item_id'] = np.nan
if 'i2i_w02' in recall_diff_road_list:
    print('start recall i2i_w02')
    i2i_w02_recall(df_train, df_train_stage, df, df_stage)
    print('end recall i2i_w02')
if 'b2b' in recall_diff_road_list:
    print('start recall b2b_recall')
    b2b_recall(df_train, df_train_stage, df, df_stage)
    print('end recall b2b_recall')
if 'i2i2i' in recall_diff_road_list:
    print('start recall i2i2i_recall')
    i2i2i_recall(df_train, df_train_stage, df, df_stage)
    print('end recall i2i2i_recall')
if 'i2i2i_new' in recall_diff_road_list:
    print('start recall i2i2i_new_recall')
    i2i2i_new_recall(df_train, df_train_stage, df, df_stage)
    print('end recall i2i2i_new_recall')
if 'i2i_w10' in recall_diff_road_list:
    print('start recall i2i_w10_recall')
    i2i_w10_recall(df_train, df_train_stage, df, df_stage)
    print('end recall i2i_w10_recall')
if 'b2b2b' in recall_diff_road_list:
    print('start recall b2b2b_recall')
    b2b2b_recall(df_train, df_train_stage, df, df_stage)
    print('end recall b2b2b_recall')
if 'i2i2b' in recall_diff_road_list:
    print('start recall i2i2b_recall')
    i2i2b_recall(df_train, df_train_stage, df, df_stage)
    print('end recall i2i2b_recall')
if 'b2b2i' in recall_diff_road_list:
    print('start recall b2b2i_recall')
    b2b2i_recall(df_train, df_train_stage, df, df_stage)
    print('end recall b2b2i_recall')
if 'b2bl2' in recall_diff_road_list:
    print('start recall b2bl2_recall')
    b2bl2_recall(df_train, df_train_stage, df, df_stage)
    print('end recall b2bl2_recall')
