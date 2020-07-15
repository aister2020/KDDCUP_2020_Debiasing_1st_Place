# -*- coding: utf-8 -*-

import numpy as np
import utils
from constants import *
def evaluate_each_phase(predictions, answers, at=50):
    list_item_degress = []
    for item_id,item_degree in answers:
        list_item_degress.append(item_degree)
        
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    
    row = 0

    for item_id,item_degree in answers:
        rank = 0
        while rank < at and predictions[row][rank] != item_id:
            rank += 1
            
        num_cases_full += 1.0
        if rank < at:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank < at:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
        
        row += 1
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([hitrate_50_full, ndcg_50_full, hitrate_50_half, ndcg_50_half], dtype=np.float32)

def evaluate_hitrate(predictions, answers, at=50):
    
    hit = 0
    for pred,ans in zip(predictions,answers):
        if ans in pred[:at]:
            hit += 1

    return hit/len(answers)


def eval_valid(prediction,items,index,stage,predict_answer,at=50):
    
    item_degree = utils.load_pickle(phase_full_item_degree_path.format(stage))
    
    df_valid = utils.load_pickle(predict_answer)
    

    prediction = np.array(prediction)
    
    shape0 = df_valid.shape[0]
    
    items = np.array(items).reshape((shape0,-1))
    
    candidate_items = items
    
    pos = df_valid['item_id'].values
    
    
    
    
    
    # shape1 = len(candidate_items)
    # split = int(len(candidate_items)/item_num)
    # if shape1%item_num != 0:
        # split += 1
    
    # shape1 = item_num*split
    
    
    rsp = prediction.reshape((shape0,-1))
    #rsp = rsp[:,:candidate_item_num]
    
    predictions = (-rsp).argsort(axis=1)
    
    r_predictions = []
    for i in range(shape0):
        r_predictions.append( candidate_items[i][ predictions[i] ] )
    
    predictions = np.array(r_predictions)
    #import pdb
    #pdb.set_trace()
    predictions1 = predictions[:,:at]

    answers = [(p,item_degree[stage][p]) for p in pos]
    scores = evaluate_each_phase(predictions1, answers, at)
    
    
    df_valid['src_index'] = np.arange(len(df_valid)) 
    df_valid_stage = utils.load_pickle(all_valid_stage_data_path.format(stage))
    df_valid_stage = df_valid_stage.merge(df_valid[['user_id','item_id','time','src_index']],on=['user_id','item_id','time'])
    
    
    df_train_stage = utils.load_pickle(all_train_stage_data_path.format(stage))
    
    clicked_item_set = df_train_stage.groupby(['user_id'])['item_id'].agg(set)
    
    stage_item_set = df_train_stage.groupby(['stage'])['item_id'].agg(set)
    
    all_stage_scores = []
    for sta,group in df_valid_stage.groupby('stage'):
        stage_preds = []
        pos = []
        
        for user_id,src_index,item_id in zip(group['user_id'],group['src_index'],group['item_id']):
            pos.append(item_id)
            preds = []
            
            stage_item = stage_item_set.loc[sta]
            clicked_item = set()
            if user_id in clicked_item_set:
                clicked_item = clicked_item_set.loc[user_id]
                
            for pr in predictions[src_index]:
                #if pr not in clicked_item:
                #    if pr in stage_item:
                preds.append(pr)
                if len(preds)>=at:
                    break
            
            stage_preds.append(preds)
            
        answers = [(p,item_degree[sta][p]) for p in pos]
        stage_scores = evaluate_each_phase(stage_preds, answers, at)
        all_stage_scores.append(stage_scores)
        
    
    return scores,all_stage_scores


def predict(prediction,items,index,stage,predict_answer,test_result,VALID):
    
    import pdb
    pdb.set_trace()
    df_predict = utils.load_pickle(predict_answer)
    
    prediction = np.array(prediction)
    
    shape0 = df_predict.shape[0]
    
    items = np.array(items).reshape((shape0,-1))[0]
    
    candidate_items = items
    
    rsp = prediction.reshape((shape0,-1))
    
    
    predictions = (-rsp).argsort(axis=1)
    pred = (-rsp)
    pred.sort(axis=1)
    pred = -pred
    
    predictions = candidate_items[predictions]
    
    
    if VALID==1:
        pkl_data = {'user_id':df_predict['user_id'].values,'time':df_predict['time'].values,'predict_score':pred[:5000],'predict_item':predictions[:5000]}
        if 'item_id' in df_predict.columns:
            pkl_data['item_id'] = df_predict['item_id'].values
            
        utils.dump_pickle(pkl_data,test_result)
        return 
    
    else:
        df_predict['src_index'] = np.arange(len(df_predict)) 
        df_train_stage = utils.load_pickle(all_train_stage_data_path.format(stage))
        
        clicked_item_set = df_train_stage.groupby(['user_id'])['item_id'].agg(set)
        
        stage_item_set = df_train_stage.groupby(['stage'])['item_id'].agg(set)
        
        all_users = []
        all_preds = []
        for sta,group in df_predict.groupby('stage'):
            
            for user_id,src_index in zip(group['user_id'],group['src_index']):
                preds = []
                
                stage_item = stage_item_set.loc[sta]
                clicked_item = set()
                if user_id in clicked_item_set:
                    clicked_item = clicked_item_set.loc[user_id]
                    
                for pr in predictions[src_index]:
                    if pr not in clicked_item:
                        if pr in stage_item:
                            preds.append(pr)
                            if len(preds)>=50:
                                break
                
                all_users.append(user_id)
                all_preds.append(preds)
                
        
        with open(test_result,'w') as file:
            for user,pred in zip(all_users,all_preds):
                file.write(str(user)+','+','.join([str(p) for p in pred])+'\n')
                    
        return
        