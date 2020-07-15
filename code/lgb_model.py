#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from constants import *
from datetime import datetime
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import utils
import os
import ndcg_tools
import math
import gc
import sys
seed = SEED
cur_stage = CUR_STAGE
version = datetime.now().strftime("%m%d%H%M%S")
print('Version: ', version)
weights = [6.5,1]
print('Now using weight: ', weights)

LR = '0.005'
load_model = ""

if len(sys.argv)>0:
    LR = sys.argv[1]
    load_model = sys.argv[2]
print('LR Status: ', LR, ' load_model: ', load_model)
def modeling(train_X, train_Y, test_X, test_Y, categoricals, mode, OPT_ROUNDS=600, weight=None):
    EARLY_STOP = 300
    OPT_ROUNDS = OPT_ROUNDS
    MAX_ROUNDS = 10000
    params = {
        'boosting': 'gbdt',
        'metric' : 'binary_logloss',
        #'metric' : 'auc',
        'objective': 'binary',
        'learning_rate': float(LR),
        'max_depth': -1,
        'min_child_samples': 20,
        'max_bin': 255,
        'subsample': 0.85,
        'subsample_freq': 10,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.001,
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'num_leaves':63,
        'seed': seed,
        'nthread': 16,
        'scale_pos_weight': 1.5
        #'is_unbalance': True,
    }
    print(f'Now Version {version}')
    if mode == 'valid':
        print('Start train and validate...')
        print('feature number:', len(train_X.columns))
        feat_cols = list(train_X.columns)
        dtrain = lgb.Dataset(data=train_X, label=train_Y, feature_name=feat_cols,weight=weight)
        dvalid = lgb.Dataset(data=test_X, label=test_Y, feature_name=feat_cols)
        model = lgb.train(params,
                          dtrain,
                          categorical_feature=categoricals,
                          num_boost_round=MAX_ROUNDS,
                          early_stopping_rounds=EARLY_STOP,
                          verbose_eval=50,
                          valid_sets=[dtrain, dvalid],
                          valid_names=['train', 'valid']
                          )
        importances = pd.DataFrame({'features':model.feature_name(),
                                'importances':model.feature_importance()})
        importances.sort_values('importances',ascending=False,inplace=True)
        importances.to_csv( (feat_imp_dir+'{}_imp.csv').format(version), index=False )
        return model
    else:
        print('Start training... Please set OPT-ROUNDS.')
        feat_cols = list(train_X.columns)
        dtrain = lgb.Dataset(data=train_X, label=train_Y, feature_name=feat_cols,weight=weight)
        print('feature number:', len(train_X.columns))
        print('feature :', train_X.columns)
        model = lgb.train(params,
                          dtrain,
                          categorical_feature=categoricals,
                          num_boost_round=OPT_ROUNDS,
                          verbose_eval=50,
                          valid_sets=[dtrain],
                          valid_names='train'
                          )
        
        importances = pd.DataFrame({'features':model.feature_name(),
                                'importances':model.feature_importance()})
        importances.sort_values('importances',ascending=False,inplace=True)
        importances.to_csv( (feat_imp_dir+'{}_imp.csv').format(version), index=False )
        
        model.save_model( lgb_model_dir+'{}.model'.format(version) )
        return model
    
def predict(test_X, model):
    print('Start Predict ...')
    print('Num of features: ', len(test_X.columns))
    print(test_X.columns)
    block_len = len(test_X)//block_num
    predicts = []
    for block_id in range(block_num):
        l = block_id * block_len
        r = (block_id+1) * block_len
        if block_id == block_num - 1:
            predict = model.predict( test_X.iloc[l:], num_iteration=model.best_iteration)
        else:
            predict = model.predict( test_X.iloc[l:r], num_iteration=model.best_iteration)
        predicts.append(predict)
    predict = np.concatenate( predicts )
    return predict
    
def get_scores(ans=None,shift=0.0,bottom=0.25,after_deal=True):
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    df_valid_stage = utils.load_pickle(all_valid_stage_data_path.format(cur_stage))


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


    old = False
    if after_deal:
        ans = ans.groupby( ['user', 'item'] )['label'].max().reset_index()
        if old:
            user_item_label = ans[ ['user','item','label'] ].values
            user2stage = df_valid_stage[ ['user_id','stage'] ]
            user2stage['user'] = user2stage['user_id']
            user2stage = user2stage.drop('user_id', axis=1)
            ans = pd.merge( ans, user2stage, how='left', on='user' )

            sta_list = []
            item_list = []
            degree_list = []

            for sta in range(cur_stage+1):
                item_degrees = phase_item_degree[sta]
                for item in item_degrees.keys():
                    sta_list.append(sta)
                    item_list.append(item)
                    degree_list.append( item_degrees[item] )
            df_degree = pd.DataFrame( {'stage':sta_list, 'item':item_list, 'degree':degree_list} )
            ans = pd.merge( ans, df_degree, how='left', on=['stage','item'] )
            phase_median = ans.groupby('stage')['degree'].median().reset_index()
            phase_median['median_degree'] = phase_median['degree']
            phase_median = phase_median.drop('degree', axis=1)
            ans = pd.merge(ans, phase_median, how='left', on ='stage')
            ans['is_rare'] = ans['degree'] <= (ans['median_degree']+shift)
        else:
            user2stage = df_valid_stage[ ['user_id','stage'] ]
            user2stage['user'] = user2stage['user_id']
            user2stage = user2stage.drop('user_id', axis=1)
            ans = pd.merge( ans, user2stage, how='left', on='user' )

            vals = ans[ ['item','stage'] ].values
            is_rare = []
            for val in vals:
                is_rare.append( phase_item_degree[ val[1] ][ val[0] ] <= phase2median[ val[1] ] )
            ans['is_rare'] = is_rare
        ans['is_rare'] = ans['is_rare'].astype('float') / bottom
        ans['is_rare'] = ans['is_rare']+1.0
        ans['label'] = ans['label'] * ans['is_rare']
    else:
        ans = ans.groupby( ['user', 'item'] )['label'].max().reset_index()

    ans['label'] = -ans['label']
    ans = ans.sort_values( by=['user','label'] )
    user2recall = ans.groupby('user')['item'].agg(list)
    user2pos = df_valid_stage[ ['user_id','item_id'] ].set_index('user_id')

    
    all_scores = []
    all_pred_items = {}
    pickup = 500
    for sta in range(cur_stage+1):
        predictions = []
        item_degree = phase_item_degree[sta]
        now_users = df_valid_stage[ df_valid_stage['stage']==sta ]['user_id'].tolist()
        answers = []
        for now_user in now_users:
            pos = user2pos.loc[now_user].values[0]
            pred = user2recall.loc[now_user]
            new_pred = []
            for j in pred:
                if len(new_pred) < pickup:
                    flag = 0
                    for k in new_pred:
                        if j == k:
                            flag = 1
                            break
                    if flag==0:
                        new_pred.append( j )
                    
            answers.append( (  pos, item_degree[ pos ] ) )
            all_pred_items[now_user] = []
            for pred in new_pred[:pickup]:
                all_pred_items[now_user].append( pred )
            predictions.append(new_pred[:50]+[0]*(50-len(new_pred)))
      
        scores = ndcg_tools.evaluate_each_phase(predictions, answers, at=50)
        all_scores.append(scores)
    utils.dump_pickle(all_pred_items, rerank_path.format(pickup, mode))
    for scores in all_scores:
        print(scores)
    print('all_scores_sum',np.array(all_scores).sum(axis=0))
    print('7_9_all_scores_sum',np.array(all_scores[-3:]).sum(axis=0))
    print('0_6_all_scores_sum',np.array(all_scores[0:7]).sum(axis=0))
    return all_scores

def get_result(ans=None,shift=0.0,bottom=0.7,after_deal=True):    
    print(f'using bottom: {bottom}')
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    df_test_stage = utils.load_pickle(online_all_test_data_path.format(cur_stage))

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
    
    old = False
    if after_deal:
        ans = ans.groupby( ['user', 'item'] )['label'].max().reset_index()
        if old:
            user_item_label = ans[ ['user','item','label'] ].values
            user2stage = df_test_stage[ ['user_id','stage'] ]
            user2stage['user'] = user2stage['user_id']
            user2stage = user2stage.drop('user_id', axis=1)
            ans = pd.merge( ans, user2stage, how='left', on='user' )

            sta_list = []
            item_list = []
            degree_list = []
            for sta in range(cur_stage+1):
                item_degrees = phase_item_degree[sta]
                for item in item_degrees.keys():
                    sta_list.append(sta)
                    item_list.append(item)
                    degree_list.append( item_degrees[item] )
            df_degree = pd.DataFrame( {'stage':sta_list, 'item':item_list, 'degree':degree_list} )
            ans = pd.merge( ans, df_degree, how='left', on=['stage','item'] )
            phase_median = ans.groupby('stage')['degree'].median().reset_index()
            phase_median['median_degree'] = phase_median['degree']
            phase_median = phase_median.drop('degree', axis=1)
            ans = pd.merge(ans, phase_median, how='left', on ='stage')
            ans['is_rare'] = ans['degree'] <= (ans['median_degree']+shift)
        else:
            user2stage = df_test_stage[ ['user_id','stage'] ]
            user2stage['user'] = user2stage['user_id']
            user2stage = user2stage.drop('user_id', axis=1)
            ans = pd.merge( ans, user2stage, how='left', on='user' )
            vals = ans[ ['item','stage'] ].values
            is_rare = []
            for val in vals:
                is_rare.append( phase_item_degree[ val[1] ][ val[0] ] <= phase2median[ val[1] ] )
            ans['is_rare'] = is_rare
        ans['is_rare'] = ans['is_rare'].astype('float') / bottom
        ans['is_rare'] = ans['is_rare']+1.0
        ans['label'] = ans['label'] * ans['is_rare']
    else:
        ans = ans.groupby( ['user', 'item'] )['label'].max().reset_index()

    ans['label'] = -ans['label']
    ans = ans.sort_values( by=['user','label'] )
    user2recall = ans.groupby('user')['item'].agg(list)

    df_train_stage = utils.load_pickle(online_all_train_stage_data_path.format(cur_stage))
    all_scores = []
    all_pred_items = {}
    pickup = 500
    predictions = {}
    for sta in range(cur_stage+1):
        now_users = df_test_stage[ df_test_stage['stage'] == sta ]['user_id'].tolist()
        df_train = df_train_stage[ df_train_stage['stage'] == sta ]
        hot_items = df_train['item_id'].value_counts().index.tolist()
        
        answers = []
        for now_user in now_users:
            pred = user2recall.loc[now_user]
            new_pred = []
            for j in pred:
                if (len(new_pred) < pickup) and (j not in new_pred):
                    new_pred.append( j )

            all_pred_items[now_user] = []
            for pred in new_pred[:pickup]:
                all_pred_items[now_user].append(pred)

            new_pred = new_pred[:50]
            for j in hot_items:
                if (len(new_pred) < 50) and (j not in new_pred):
                    new_pred.append( j )
                    
            predictions[now_user] = new_pred

    utils.dump_pickle(all_pred_items, rerank_path.format(pickup, mode))
    #check
    
    with open(prediction_result+f'{version}_{LR}_result.csv','w') as file:
        for idx,user in enumerate(predictions.keys()):
            file.write(str(user)+','+','.join([str(p) for p in predictions[user]])+'\n')

def debug_scores(ans,shift=0.0, bottom=0.25,after_deal=True):
    
    #1)
    #count_info = ans.groupby( ['user', 'item'] )['label'].count().reset_index()
    #def func(s):
    #    return math.log(1 + s)
    #count_info['label'] = count_info['label'].apply( func )
    #ans = ans.groupby( ['user', 'item'] )['label'].sum().reset_index()
    #ans['label'] = ans['label'] / count_info['label']
    
    #2)
    #ans = ans.groupby( ['user', 'item'] )['label'].max().reset_index()

    #3)
    #ans = ans.groupby( ['user', 'item'] )['label'].mean().reset_index()

    #4)
    #ans = ans.groupby( ['user', 'item'] )['label'].sum().reset_index()
    
    #5)
    
    df_valid_stage = utils.load_pickle(all_valid_stage_data_path.format(cur_stage))
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

    old = False
    if after_deal:
        ans = ans.groupby( ['user', 'item'] )['label'].max().reset_index()
        if old:
            user_item_label = ans[ ['user','item','label'] ].values
            user2stage = df_valid_stage[ ['user_id','stage'] ]
            user2stage['user'] = user2stage['user_id']
            user2stage = user2stage.drop('user_id', axis=1)
            ans = pd.merge( ans, user2stage, how='left', on='user' )

            sta_list = []
            item_list = []
            degree_list = []

            for sta in range(cur_stage+1):
                item_degrees = phase_item_degree[sta]
                for item in item_degrees.keys():
                    sta_list.append(sta)
                    item_list.append(item)
                    degree_list.append( item_degrees[item] )
            df_degree = pd.DataFrame( {'stage':sta_list, 'item':item_list, 'degree':degree_list} )
            ans = pd.merge( ans, df_degree, how='left', on=['stage','item'] )
            phase_median = ans.groupby('stage')['degree'].median().reset_index()
            phase_median['median_degree'] = phase_median['degree']
            phase_median = phase_median.drop('degree', axis=1)
            ans = pd.merge(ans, phase_median, how='left', on ='stage')
            ans['is_rare'] = ans['degree'] <= (ans['median_degree']+shift)
        else:
            user2stage = df_valid_stage[ ['user_id','stage'] ]
            user2stage['user'] = user2stage['user_id']
            user2stage = user2stage.drop('user_id', axis=1)
            ans = pd.merge( ans, user2stage, how='left', on='user' )
            vals = ans[ ['item','stage'] ].values
            is_rare = []
            for val in vals:
                is_rare.append( phase_item_degree[ val[1] ][ val[0] ] <= phase2median[ val[1] ] )
            ans['is_rare'] = is_rare
        ans['is_rare'] = ans['is_rare'].astype('float') / bottom
        ans['is_rare'] = ans['is_rare']+1.0
        ans['label'] = ans['label'] * ans['is_rare']
    else:
        ans = ans.groupby( ['user', 'item'] )['label'].max().reset_index()

    ans['label'] = -ans['label']
    ans = ans.sort_values( by=['user','label'] )
    user2recall = ans.groupby('user')['item'].agg(list)
    
    user2pos = df_valid_stage[ ['user_id','item_id'] ].set_index('user_id')

    all_scores = []
    for sta in range(cur_stage+1):
        predictions = []
        item_degree = phase_item_degree[sta]
        now_users = df_valid_stage[ df_valid_stage['stage']==sta ]['user_id'].tolist()
        answers = []
        for now_user in now_users:
            pos = user2pos.loc[now_user].values[0]
            pred = user2recall.loc[now_user]
            new_pred = []
            for j in pred:
                if len(new_pred) < 50:
                    flag = 0
                    for k in new_pred:
                        if j == k:
                            flag = 1
                            break
                    if flag==0:
                        new_pred.append( j )
                    
            answers.append( (  pos, item_degree[ pos ] ) )
            predictions.append(new_pred+[0]*(50-len(new_pred)))
      
        scores = ndcg_tools.evaluate_each_phase(predictions, answers, at=50)
        all_scores.append(scores)
    for scores in all_scores:
        print(scores)
    print('all_scores_sum',np.array(all_scores).sum(axis=0))
    print('7_9_all_scores_sum',np.array(all_scores[-3:]).sum(axis=0))
    print('0_6_all_scores_sum',np.array(all_scores[0:7]).sum(axis=0))
    return all_scores

def calculate_user2degree( ans, df_valid_stage, phase_item_degree, top=50 ):
    count_info = ans.groupby( ['user', 'item'] )['label'].count().reset_index()
    def func(s):
        return s-0.2
    count_info['label'] = count_info['label'].apply( func )
    ans = ans.groupby( ['user', 'item'] )['label'].sum().reset_index()
    ans['label'] = ans['label'] / count_info['label']
    ans['label'] = -ans['label']
    ans = ans.sort_values( by=['user','label'] )
    user2recall = ans.groupby('user')['item'].agg(list)
    
    user2pos = df_valid_stage[ ['user_id','item_id'] ].set_index('user_id')

    all_scores = []
    user2degree = {}
    user2stage = {}
    for sta in range(cur_stage+1):
        predictions = []
        item_degree = phase_item_degree[sta]
        now_users = df_valid_stage[ df_valid_stage['stage']==sta ]['user_id'].tolist()
        answers = []
        for now_user in now_users:
            pos = user2pos.loc[now_user].values[0]
            pred = user2recall.loc[now_user]
            new_pred = []
            for j in pred:
                if len(new_pred) < 50:
                    flag = 0
                    for k in new_pred:
                        if j == k:
                            flag = 1
                            break
                    if flag==0:
                        new_pred.append( j )
                    
            answers.append( (  pos, item_degree[ pos ] ) )
            degrees = []
            for j in new_pred[:50]:
                if j in item_degree:
                    degrees.append( item_degree[j] )
                else:
                    print('no')
                    print(1/0)
            user2degree[now_user] = np.mean( degrees )
            user2stage[now_user] = sta
            predictions.append(new_pred+[0]*(50-len(new_pred)))

    return user2degree, user2stage

def model_merge_scores():
    full = utils.load_pickle( lgb_ans_dir+ ('{}_{}_{}_ans.pkl').format('0529094924', mode, cur_stage ) )
    rare = utils.load_pickle( lgb_ans_dir+ ('{}_{}_{}_ans.pkl').format('0529095003', mode, cur_stage ) )
    df_valid_stage = utils.load_pickle(all_valid_stage_data_path.format(cur_stage))
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage))
    use_gt = False
    if use_gt:
        user2degree = {}
        for i in range( df_valid_stage.shape[0] ):
            r = df_valid_stage.iloc[i]
            user2degree[ r['user_id'] ] = phase_item_degree[r['stage']][ r['item_id'] ]  

        data = df_valid_stage.copy()
        data['degree'] = data['user_id'].map( user2degree )
        data['median_degree'] = data['stage'].map( data.groupby('stage')['degree'].quantile(0.5) )
        data['is_rare'] = data['degree']<=data['median_degree']
        data['user'] = data['user_id']
        full = pd.merge( full, data[ ['user', 'is_rare'] ], how='left', on='user' )
        rare = pd.merge( rare, data[ ['user', 'is_rare'] ], how='left', on='user' )
        ans = full.copy()
        #ans = ans.groupby( ['user', 'item'] )['label'].quantile(0.25).reset_index()
        ans['label'] = full['label']*(1-full['is_rare']) + rare['label']*rare['is_rare']
    else: 
        user2degree, user2stage = calculate_user2degree( rare.copy(), df_valid_stage, phase_item_degree, 50 ) 
        data = pd.concat( [pd.Series(user2degree), pd.Series(user2stage)], axis=1 ).reset_index()
        data.columns = ['user','degree','stage']
        medians = data.groupby('stage')['degree'].quantile(0.75).reset_index()
        medians.columns = ['stage','mid_degree']
        
        full['stage'] = full['user'].map( user2stage )
        full = pd.merge( full, medians, how='left', on='stage' )
        full = pd.merge( full, data[ ['user','degree'] ], how='left', on='user' )
        full['is_rare'] = full['degree'] > full['mid_degree']
        #full['is_rare'] = 

        rare['stage'] = rare['user'].map( user2stage )
        rare = pd.merge( rare, medians, how='left', on='stage' )
        rare = pd.merge( rare, data[ ['user','degree'] ], how='left', on='user' )
        rare['is_rare'] = rare['degree'] <= rare['mid_degree']
        ans = full.copy()
        #ans = ans.groupby( ['user', 'item'] )['label'].quantile(0.25).reset_index()
        ans['label'] = full['label']*full['is_rare'] + rare['label']*rare['is_rare']

    
    '''
    count_info = ans.groupby( ['user', 'item'] )['label'].count().reset_index()
    def func(s):
        return s-0.2
    count_info['label'] = count_info['label'].apply( func )
    ans = ans.groupby( ['user', 'item'] )['label'].sum().reset_index()
    ans['label'] = ans['label'] / count_info['label']
    '''
    #import pdb
    #pdb.set_trace()

    '''
    user2stage = df_valid_stage[ ['user_id','stage'] ]
    user2stage['user'] = user2stage['user_id']
    user2stage = user2stage.drop('user_id', axis=1)
    ans = pd.merge( ans, user2stage, how='left', on='user' )

    user_item_label = ans[ ['user','item','label'] ].values
    
    sta_list = []
    item_list = []
    degree_list = []

    for sta in range(cur_stage+1):
        item_degrees = phase_item_degree[sta]
        for item in item_degrees.keys():
            sta_list.append(sta)
            item_list.append(item)
            degree_list.append( item_degrees[item] )
    df_degree = pd.DataFrame( {'stage':sta_list, 'item':item_list, 'degree':degree_list} )
    ans = pd.merge( ans, df_degree, how='left', on=['stage','item'] )
    phase_median = ans.groupby('stage')['degree'].median().reset_index()
    phase_median['median_degree'] = phase_median['degree']
    phase_median = phase_median.drop('degree', axis=1)
    ans = pd.merge(ans, phase_median, how='left', on ='stage')
    ans['is_rare'] = ans['degree'] <= (ans['median_degree']+shift)
    ans['is_rare'] = ans['is_rare'].astype('float') / bottom
    ans['is_rare'] = ans['is_rare']+1.0
    ans['label'] = ans['label'] * ans['is_rare']
    #import pdb
    #pdb.set_trace()
    '''


    
    ans['label'] = -ans['label']
    ans = ans.sort_values( by=['user','label'] )
    user2recall = ans.groupby('user')['item'].agg(list)
    
    user2pos = df_valid_stage[ ['user_id','item_id'] ].set_index('user_id')

    all_scores = []
    for sta in range(cur_stage+1):
        predictions = []
        item_degree = phase_item_degree[sta]
        now_users = df_valid_stage[ df_valid_stage['stage']==sta ]['user_id'].tolist()
        answers = []
        for now_user in now_users:
            pos = user2pos.loc[now_user].values[0]
            pred = user2recall.loc[now_user]
            new_pred = []
            for j in pred:
                if len(new_pred) < 50:
                    flag = 0
                    for k in new_pred:
                        if j == k:
                            flag = 1
                            break
                    if flag==0:
                        new_pred.append( j )
                    
            answers.append( (  pos, item_degree[ pos ] ) )
            predictions.append(new_pred+[0]*(50-len(new_pred)))
      
        scores = ndcg_tools.evaluate_each_phase(predictions, answers, at=50)
        all_scores.append(scores)
    for scores in all_scores:
        print(scores)
    print('all_scores_sum',np.array(all_scores).sum(axis=0))
    
    return all_scores

if __name__ == '__main__':
    mode = cur_mode
    used_recall_source = cur_used_recall_source
    sum_mode = 'nosum'
    used_recall_source = used_recall_source+'-'+sum_mode
    print( f'Recall Source Use {used_recall_source} now mode {mode}')
    

    '''
    #参数搜索。
    shifts = [ 0 ]
    #biass = [ 0.005, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0 ]
    
    #biass = [ 0.15, 0.2, 0.25 ]
    #biass = [ 0.35, 0.4, 0.45 ]
    #biass = [ 0.5, 0.55, 0.6 ]
    #biass = [ 0.65, 0.7, 0.8 ]
    #biass = [ 0.85, 0.9 , 1.0 ]
    #baiss = [ 1.25, 1.5 ]
    #6.5 scale: 0606091051
    weight2version = { 5.5:'0606090851', 6.5:'0605205128', 8.5:'0606081613' } 
    weight2version = { 10.5:'0606081643', 12.5:'0606081705' }
    weight2version = { 14.5:'0606081729', 16:'0606090940' }
    weight2version = { 6.5:'0607185706' }
    for weight in weight2version.keys():
        datas = []
        for block_id in range(block_num):
            datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_ans.pkl'.format(weight2version[weight], 'valid', cur_stage, block_id ) ) )
        data = pd.concat( datas )
        ans = data
        for shift in shifts:
            for bias in biass:
                print('====== Shift: ', shift, ' Bias: ', bias, ' wegiht: ', weight, 'version: ',weight2version[weight] ,'========')
                debug_scores(ans, shift, bias)
                print('=====================================================')
    print(1/0)
    '''



    #'0603205234' #倒数第三个
    #'0602121256' #倒数第二个
    #'0601151925' #倒数第一个

    '''
    #模型融合online
    #ans1 = utils.load_pickle( (lgb_ans_dir+'{}_{}_{}_ans.pkl').format('0602121256', 'test', cur_stage) )
    #ans2 = utils.load_pickle( (lgb_ans_dir+'{}_{}_{}_ans.pkl').format('0601151925', 'test', cur_stage) )
    #ans3 = utils.load_pickle( (lgb_ans_dir+'{}_{}_{}_ans.pkl').format('0603205234', 'test', cur_stage) )
    #ans = utils.load_pickle( (lgb_ans_dir+'{}_{}_{}_ans.pkl').format('0601151925', 'test', cur_stage) )
    #ans = utils.load_pickle( lgb_ans_dir+ ('{}_{}_{}_ans.pkl').format(version, mode, cur_stage ) ) 
    #import pdb
    #pdb.set_trace()
    #ans['label'] = 0.4*ans1['label'] + 0.5*ans2['label'] + 0.1*ans3['label'] 
    #ans = utils.load_pickle( lgb_ans_dir+ ('{}_{}_{}_ans.pkl').format('0607102652', mode, cur_stage ) ) #0607
    #ans = utils.load_pickle( lgb_ans_dir+ ('{}_{}_{}_ans.pkl').format('0607102652', mode, cur_stage ) ) 
    ans = utils.load_pickle( lgb_ans_dir+ ('{}_{}_{}_ans.pkl').format('0604143430', 'test', cur_stage ) ) 
    #ans = utils.load_pickle( lgb_ans_dir+ ('{}_{}_{}_ans.pkl').format('0604113916', 'test', cur_stage ) ) 
    get_result(ans=ans,shift=0.0,bottom=0.25,after_deal=True)
    #get_result(ans=ans)
    print(1/0)
    '''
    
    '''
    with open(prediction_result+f'{version}_result.csv','w') as file:
        for idx,user in enumerate(predictions.keys()):
            if idx in mask_indexs:
                file.write(str(user)+','+','.join([str(p) for p in mask_preds])+'\n')
            else:
                file.write(str(user)+','+','.join([str(p) for p in predictions[user]])+'\n')
    '''

    '''
    ans1 = pd.read_csv('../prediction_result/0603155052_result.csv',header=None)
    ans2 = pd.read_csv('../prediction_result/0603152047_result.csv',header=None)
    df_test_stage = utils.load_pickle(online_all_test_data_path.format(cur_stage))
    vals = df_test_stage[ ['user_id','stage'] ].values
    user2stage = {}
    for val in vals:
        user2stage[val[0]] = val[1]
    ans1['stage'] = ans1[0].map(user2stage)
    ans2['stage'] = ans2[0].map(user2stage)
    ans = pd.concat( [ans1[ans1['stage']!=6] ,ans2[ans2['stage']==6]] ).reset_index(drop=True)
    ans = ans.drop( 'stage', axis=1 )
    ans.to_csv('huangjianqiang.csv',index=False,header=None)
    '''

    '''
    ans = utils.load_pickle( (lgb_ans_dir+'{}_{}_{}_ans.pkl').format('0604113916', 'test', cur_stage) )
    # all
    #ans2 = utils.load_pickle( (lgb_ans_dir+'{}_{}_{}_ans.pkl').format('0604145925', 'test', cur_stage) )
    # 最后一个阶段
    #ans = pd.concat( [ ans1[ ans1['stage']!=6 ], ans2[ ans2['stage']==6 ] ] ).reset_index(drop=True)
    #import pdb
    #pdb.set_trace()
    
    get_result(ans=ans)
    print(1/0)
    '''

    '''
    #生成建模指标的数据
    version = '0602104358'
    ans = utils.load_pickle( (lgb_ans_dir+'{}_{}_{}_ans.pkl').format(version, 'valid', cur_stage) )
    tdata = pd.pivot_table(ans,index=['user','item'],columns=['recall_road','recall_type'],values=['label'],fill_value=np.nan)
    block_len = len(tdata)//block_num
    for block_id in range(block_num):
        l = block_id * block_len
        r = (block_id+1) * block_len
        if block_id == block_num - 1:
            utils.dump_pickle( tdata.iloc[l:], (lgb_pivot_dir+'{}_{}_{}_{}.pkl').format(version, 'valid', cur_stage,block_id) )
        else:
            utils.dump_pickle( tdata.iloc[l:r], (lgb_pivot_dir+'{}_{}_{}_{}.pkl').format(version, 'valid', cur_stage,block_id) )
    print(1/0)
    '''


    drop_list = [ 'label','left_items_list', 'right_items_list', 'left_times_list',
            'right_times_list','item','rank_weight','item_sum_rank_weight',
            'item_mean_rank_weight','user','time','recall_type',
            'sim_weight',
            ]


    categorical_list = [ 'stage' ]

    

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
    
    
    OPT_ROUNDS = 1000
    if mode == 'test':
        if load_model=="":
            train_datas = []
            for block_id in range(merge_block_num):
                train_datas.append( utils.load_pickle( lgb_model_pkl.format( used_recall_source,'valid',cur_stage,block_id ) ) )
                print(f'reading train {block_id}')
            train_data = pd.concat( train_datas )
            del train_datas
            gc.collect()

            
            test_datas = []
            for block_id in range(merge_block_num):
                test_datas.append( utils.load_pickle( lgb_model_pkl.format( used_recall_source,'test',cur_stage,block_id ) ) )
                print(f'reading test {block_id}')
            test_data = pd.concat( test_datas )
            del test_datas
            gc.collect()

            if train_data.shape[1] != test_data.shape[1]:
                print(f'训练集长度和测试集特征不match, train:{train_data.shape[1]}, test:{test_data.shape[1]}')
                print(1/0)
            del test_data
            gc.collect()
            print('OK testing')
        

            ts1 = train_data.shape
            all_users = train_data['user'].unique()
            t = train_data.groupby(['user'])['label'].any()
            train_users = list( t[t].index )
            train_index = train_data[ train_data['user'].isin(train_users) ].index
            train_data = train_data.loc[ train_index ]
            ts2 = train_data.shape
            print( list( train_data.columns ) )
            print(f'Before train data shape {ts1} After pos user filter, now train data shape: {ts2}')

            median = train_data['stage'].map(phase2median).values
                
            phase_item_degree = np.zeros(len(train_data))
            for i,(item_stage,item_id,label) in enumerate(zip(train_data['stage'].values,train_data['item'].values,train_data['label'].values)):
                if label == 1:
                    phase_item_degree[i] = phase2valid_item_degree[(item_stage,item_id)]
                else:
                    phase_item_degree[i] = np.nan
            weight = np.ones(len(train_data))
            
            weight[phase_item_degree <= median] = weights[0]
            weight[phase_item_degree > median] = weights[1]

            train_Y = train_data['label']
            train_X = train_data.drop( drop_list, axis=1 ) 


            model = modeling(train_X, train_Y, None, None, categorical_list, 'test', OPT_ROUNDS, weight)
            del train_X, train_Y, train_data
            gc.collect()
        else:
            model = lgb.Booster(model_file=lgb_model_dir+f'{load_model}.model')

        test_datas = []
        for block_id in range(merge_block_num):
            test_datas.append( utils.load_pickle( lgb_model_pkl.format( used_recall_source,'test',cur_stage,block_id ) ) )
        test_data = pd.concat( test_datas )
        del test_datas
        gc.collect()
        test_Y = test_data['label']
        test_X =  test_data.drop( drop_list, axis=1 ) 
        ans = test_data[ ['user','item','time','stage'] ]

        result = predict(test_X, model)
        ans['label'] = result
        del test_X, test_Y, test_data
        gc.collect()

        if load_model == "":
            pass
        else:
            version = load_model
        block_len = len(ans)//block_num
        for block_id in range(block_num):
            l = block_id * block_len
            r = (block_id+1) * block_len
            print('saving answer block: ', block_id)
            if block_id == block_num - 1:
                utils.dump_pickle( ans.iloc[l:], (lgb_ans_dir+'{}_{}_{}_{}_{}_ans.pkl').format(version, mode, cur_stage, block_id, LR) )
            else:
                utils.dump_pickle( ans.iloc[l:r], (lgb_ans_dir+'{}_{}_{}_{}_{}_ans.pkl').format(version, mode, cur_stage, block_id, LR) )
        
        if load_model == "":
            result = get_result(ans)
        else:
            pass
        
    if mode == 'valid':
        datas = []
        for block_id in range(merge_block_num):
            datas.append( utils.load_pickle( lgb_model_pkl.format( used_recall_source,mode,cur_stage,block_id ) ) )
        data = pd.concat( datas )
        del datas
        gc.collect()
        print('now all data_shape: ', data.shape)
        '''
        user_feat = utils.load_pickle( user_feat_pkl )
        user_feat['user'] = user_feat['user_id']
        user_feat = user_feat.drop('user_id',axis=1)
        user_feat = user_feat.loc[ user_feat['user'].drop_duplicates().index ]
        data = pd.merge( data,user_feat,on='user',how='left' )
        '''

        ans = data[ ['user','item','time','stage'] ]
        ans_tmp = ans.copy()
        all_users = data['user'].unique()
        t = data.groupby(['user'])['label'].any()
        has_pos_users = list( t[t].index )

        kfold = KFold(n_splits=3, shuffle=True, random_state=2020)                  
        index = kfold.split(X=all_users)
        for train_users_index, test_users_index in index:
            train_users = all_users[ train_users_index ]
            train_users = list( set(train_users)&set(has_pos_users) )
            train_index = data[ data['user'].isin(train_users) ].index
            test_users = all_users[ test_users_index ]
            test_index = data[ data['user'].isin(test_users) ].index
            train_data = data.loc[ train_index ]
            '''
            hjq full rare weight
            '''
            users = set()
            stage2degree = {sta:[] for sta in range(cur_stage+1)}
            phase_item_degree = np.zeros(len(train_data))
            for i, (user, item_stage, item_id, label) in enumerate(zip(train_data['user'], train_data['stage'].values,
                                                                       train_data['item'].values, train_data['label'].values)):
                if label == 1:
                    phase_item_degree[i] = phase2valid_item_degree[(item_stage,item_id)]
                    if user not in users:
                        users.add(user)
                        stage2degree[item_stage].append(phase_item_degree[i])
                else:
                    phase_item_degree[i] = np.nan
            
            phase2median = {}
            for sta in range(cur_stage+1):
                list_item_degress = stage2degree[sta]
                list_item_degress.sort()
                median_item_degree = list_item_degress[len(list_item_degress) // 2]
                phase2median[sta] = median_item_degree
            
            median = train_data['stage'].map(phase2median).values
            weight = np.ones(len(train_data))
            
            weight[phase_item_degree <= median] = weights[0]
            weight[phase_item_degree > median] = weights[1]
            

            train_X = train_data.drop( drop_list, axis=1 )
            train_Y = train_data['label']

            test_data = data.loc[ test_index ]
            test_X = test_data.drop( drop_list, axis=1 )
            test_Y = test_data['label']

            #model = modeling(train_X, train_Y, test_X, test_Y, categorical_list, 'valid', OPT_ROUNDS)
            #print('Using bast iteration: ', model.best_iteration)
            #result = predict(test_X, model)
            model = modeling(train_X, train_Y, test_X, test_Y, categorical_list, 'test', OPT_ROUNDS, weight)
            result = predict(test_X, model)
            ans_block = ans_tmp.copy()
            ans_block.loc[ test_index, 'label' ] = result
            get_scores(ans=ans_block.copy(),shift=0.0,bottom=0.25,after_deal=False)
            get_scores(ans=ans_block.copy(),shift=0.0,bottom=0.25,after_deal=True)

            ans.loc[ test_index , 'label' ] = result
        
        block_len = len(ans)//block_num
        for block_id in range(block_num):
            l = block_id * block_len
            r = (block_id+1) * block_len
            print('saving answer block: ', block_id)
            if block_id == block_num - 1:
                utils.dump_pickle( ans.iloc[l:], (lgb_ans_dir+'{}_{}_{}_{}_ans.pkl').format(version, mode, cur_stage, block_id) )
            else:
                utils.dump_pickle( ans.iloc[l:r], (lgb_ans_dir+'{}_{}_{}_{}_ans.pkl').format(version, mode, cur_stage, block_id) )

        get_scores(ans=ans.copy(),shift=0.0,bottom=0.25,after_deal=False)
        get_scores(ans=ans.copy(),shift=0.0,bottom=0.25,after_deal=True)