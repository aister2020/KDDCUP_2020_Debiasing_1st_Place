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

def get_scores(ans=None,shift=0.0,bottom=0.25,after_deal=True,save_version=None):
    print(f'using bottom: {bottom}')
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
    save_predictions = {}
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
            save_predictions[now_user] = new_pred[:50]+[0]*(50-len(new_pred))
        scores = ndcg_tools.evaluate_each_phase(predictions, answers, at=50)
        all_scores.append(scores)
    utils.dump_pickle(all_pred_items, rerank_path.format(pickup, mode))

    if save_version is None:
        save_version = version

    with open(prediction_result+f'{save_version}_result_{bottom}_tmp_valid.csv','w') as file:
        for idx,user in enumerate(save_predictions.keys()):
            file.write(str(user)+','+','.join([str(p) for p in save_predictions[user]])+'\n')
    for scores in all_scores:
        print(scores)
    print('all_scores_sum',np.array(all_scores).sum(axis=0))
    print('7_9_all_scores_sum',np.array(all_scores[-3:]).sum(axis=0))
    print('0_6_all_scores_sum',np.array(all_scores[0:7]).sum(axis=0))
    return all_scores

def get_result(ans=None,shift=0.0,bottom=0.7,after_deal=True,save_version=None):
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
    
    '''
    all_users = [ user for user in predictions.keys()]
    np.random.seed(2020)
    mask_indexs = np.random.choice(np.arange(len(all_users)),int(len(all_users)),replace=False)
    mask_indexs = set(mask_indexs)
    mask_preds = [200000+i for i in range(50)]
    
    
    for user in predictions.keys():
        if len(set(predictions[user])) != 50:
            print('no')
            print(1/0)
    '''
    with open(prediction_result+f'{save_version}.csv','w') as file:
        for idx,user in enumerate(predictions.keys()):
            file.write(str(user)+','+','.join([str(p) for p in predictions[user]])+'\n')

def cal_score(ans):
    phase_item_degree = utils.load_pickle(phase_full_item_degree_path.format(cur_stage)) 
    df_valid_stage = utils.load_pickle(all_valid_stage_data_path.format(cur_stage))

    ans = ans.set_index(0)
    user2pos = df_valid_stage[ ['user_id','item_id'] ].set_index('user_id')
    all_scores = []
    for sta in range(cur_stage+1):
        predictions = []
        item_degree = phase_item_degree[sta]
        now_users = df_valid_stage[ df_valid_stage['stage']==sta ]['user_id'].tolist()
        answers = []
        for now_user in now_users:
            pos = user2pos.loc[now_user].values[0]
            new_pred = ans.loc[now_user].tolist()                    
            answers.append( (  pos, item_degree[ pos ] ) )
            predictions.append( new_pred )

        scores = ndcg_tools.evaluate_each_phase(predictions, answers, at=50)
        all_scores.append(scores)

    for scores in all_scores:
        print(scores)
    print('all_scores_sum',np.array(all_scores).sum(axis=0))
    print('7_9_all_scores_sum',np.array(all_scores[-3:]).sum(axis=0))
    print('0_6_all_scores_sum',np.array(all_scores[0:7]).sum(axis=0))

def data_read(ver, sta='0.005'):
    datas = []
    for block_id in range(block_num):
        if sta == '0.005':
            datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_0.005_ans.pkl'.format(ver, 'test', cur_stage, block_id ) ) )
        else:
            print(1/0)
            datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_ans.pkl'.format(ver, 'test', cur_stage, block_id ) ) )
    data = pd.concat( datas )
    return data

if __name__ == '__main__':
    mode = cur_mode
    used_recall_source = cur_used_recall_source
    sum_mode = 'nosum'
    used_recall_source = used_recall_source+'-'+sum_mode
    print( f'Recall Source Use {used_recall_source} now mode {mode}')

    '''
    #model1 = lgb.Booster(model_file=lgb_model_dir+'0608185502.model')
    #model2 = lgb.Booster(model_file=lgb_model_dir+'0611052612.model')

    ans1 = data_read('0608185502', sta="")
    ans2 = data_read('0611052612', sta='0.005')
    ans3 = data_read('0611111606', sta='0.005')
    import pdb
    pdb.set_trace()
    '''

    online = True
    ensemble_mode = 'no-ensemble'
    if len(sys.argv)>0:
        ensemble_mode = sys.argv[1]

    if online == False:
        #带有坚强的特征的。
        #big:0608081804
        #fut1:0608081755
        #fut4:0608075849
        versions = { 'big':'0608081804', 'fut1':'0608081755', 'fut4':'0608075849' }
        
        #save
        '''
        for name in versions:
            datas = []
            for block_id in range(block_num):
                datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_ans.pkl'.format(versions[name], 'valid', cur_stage, block_id ) ) )
            data = pd.concat( datas )
            get_scores(ans=data,shift=0.0,bottom=0.25,after_deal=True,save_version=versions[name])
            get_scores(ans=data,shift=0.0,bottom=0.7,after_deal=True,save_version=versions[name])
        '''
        #using
        anss = {}
        for name in versions:
            data = pd.read_csv(prediction_result+f'{versions[name]}_result_0.7_tmp_valid.csv',header=None)
            anss[name] = data
        df_valid_stage = utils.load_pickle(all_valid_stage_data_path.format(cur_stage))
        tdata = df_valid_stage.groupby('user_id').first()
        user2stage = dict( tdata['stage'] )
        users = anss['big'][0].tolist()
        ans = []
        for i in range(len(users)):
            stage = user2stage[ users[i] ]
            if stage==9 :
                ans.append( anss['fut1'].iloc[i] )
            elif stage==8 :
                ans.append( anss['fut4'].iloc[i] )
            else:
                ans.append( anss['big'].iloc[i] )
        ans = pd.concat(ans,axis=1).T
        cal_score(ans)

        '''
        datas = []
        for block_id in range(merge_block_num):
            datas.append( utils.load_pickle( lgb_model_pkl.format( used_recall_source,mode,cur_stage,block_id ) ) )
        data = pd.concat( datas )
        del datas
        gc.collect()
        valid_data = data
        feat = valid_data[ ['user','left_items_list','right_items_list','stage'] ]
        def func(s):
            return len(s)
        tdata = feat.groupby('user').first()
        user2stage = dict( tdata['stage'] )
        tdata['left_items_list_len'] = tdata['left_items_list'].apply( func )
        tdata['right_items_list_len'] = tdata['right_items_list'].apply( func )
        user2right_len = dict( tdata['right_items_list_len'] )
        
        utils.dump_pickle( user2right_len, pkl_dir + 'user2right_len.pkl' )
        utils.dump_pickle( user2stage, pkl_dir + 'user2stage.pkl' )
        print(1/0)
        '''
        '''
        user2right_len = utils.load_pickle( pkl_dir + 'user2right_len.pkl' )
        user2stage = utils.load_pickle( pkl_dir + 'user2stage.pkl' )
        users = anss['fut0'][0].tolist()
        ans = []
        for i in range(len(users)):
            right_len = user2right_len[ users[i] ]
            stage = user2stage[ users[i] ]
            if stage!=7 :
                ans.append( anss['fut4'].iloc[i] )
            #elif right_len==1:
            #    ans.append( anss['fut1'].iloc[i] )
            #elif right_len<=4:
            #    ans.append( anss['fut4'].iloc[i] )
            #elif right_len<=8:
            #    ans.append( anss['fut8'].iloc[i] )
            else:
                ans.append( anss['big'].iloc[i] )
        ans = pd.concat(ans,axis=1).T
        cal_score(ans)
        import pdb
        pdb.set_trace()
        '''

        '''
        datas = []
        for block_id in range(merge_block_num):
            datas.append( utils.load_pickle( lgb_model_pkl.format( used_recall_source,mode,cur_stage,block_id ) ) )
        data = pd.concat( datas )
        del datas
        gc.collect()
        valid_data = data
        feat = valid_data[ ['user','left_items_list','right_items_list','stage'] ]
        def func(s):
            return len(s)
        tdata = feat.groupby('user').first()
        tdata['left_items_list_len'] = tdata['left_items_list'].apply( func )
        tdata['right_items_list_len'] = tdata['right_items_list'].apply( func )
        user2right_len = dict( tdata['right_items_list_len'] )
        utils.dump_pickle( user2right_len, pkl_dir + 'user2right_len.pkl' )
        '''

        '''
        #参数搜索。
        shifts = [ 0 ]
        biass = [ 0.005, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0 ]
        biass = [ 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7 ]
        #6.5 scale: 0606091051
        weight2version = { 5.5:'0606090851', 6.5:'0605205128', 8.5:'0606081613' } 
        weight2version = { 10.5:'0606081643', 12.5:'0606081705' }
        weight2version = { 14.5:'0606081729', 16:'0606090940' }
        weight2version = { 6.5:'0605205128' }
        for weight in weight2version.keys():
            ans = utils.load_pickle( lgb_ans_dir+ ('{}_{}_{}_ans.pkl').format(weight2version[weight], 'valid', cur_stage ) ) 
            for shift in shifts:
                for bias in biass:
                    print('====== Shift: ', shift, ' Bias: ', bias, ' wegiht: ', weight, 'version: ',weight2version[weight] ,'========')
                    debug_scores(ans, shift, bias)
                    print('=====================================================')
        print(1/0)
        '''

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
        pass
    else:
        #v1
        #big:0608193140
        #fut1:0608185502
        #fut4:0608200356
        #big2:0610120057
        #fut12:0610153711
        #fut42:0610125125

        #v2
        #big:611052609
        #fut1:611052612
        #fut4:611052613
        #big2:611052753
        #fut12:611052638
        #fut42:611052706

        #check
        '''
        v1anss = {}
        v1versions = {
        'big':'0608193140',
        'fut1':'0608185502',
        'fut4':'0608200356',
        'big2':'0610120057',
        'fut12':'0610153711',
        'fut42':'0610125125'}
        for name in v1versions:
            datas = []
            for block_id in range(block_num):
                if not name.endswith('2'):
                    datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_ans.pkl'.format(v1versions[name], 'test', cur_stage, block_id ) ) )
                else:
                    datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_0.005_ans.pkl'.format(v1versions[name], 'test', cur_stage, block_id ) ) )
            data = pd.concat( datas )
            v1anss[name] = data

        v2anss = {}
        v2versions = {
            'big':'0611052609',
            'fut1':'0611052612',
            'fut4':'0611052613',
            'big2':'0611052753',
            'fut12':'0611052638',
            'fut42':'0611052706'}
        for name in v2versions:
            datas = []
            for block_id in range(block_num):
                datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_0.005_ans.pkl'.format(v2versions[name], 'test', cur_stage, block_id ) ) )
            data = pd.concat( datas )
            v2anss[name] = data
        

        for name in v1versions:
            t = v1anss[name] - v2anss[name]
            print( (t>1e-7).sum() )
        import pdb
        pdb.set_trace()
        '''
        #save
        '''
        v1versions = {
            'big':'0608193140',
            'fut1':'0608185502',
            'fut4':'0608200356',
            'big2':'0610120057',
            'fut12':'0610153711',
            'fut42':'0610125125'}
        v2versions = {
            'big':'0611052609',
            'fut1':'0611052612',
            'fut4':'0611052613',
            'big2':'0611052753',
            'fut12':'0611052638',
            'fut42':'0611052706'}
        for name in v1versions:
            datas = []
            for block_id in range(block_num):
                if not name.endswith('2'):
                    datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_ans.pkl'.format(v1versions[name], 'test', cur_stage, block_id ) ) )
                else:
                    datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_0.005_ans.pkl'.format(v1versions[name], 'test', cur_stage, block_id ) ) )
            data = pd.concat( datas )
            get_result(ans=data,shift=0.0,bottom=0.7,after_deal=True,save_version='v1_'+name)

        for name in v2versions:
            datas = []
            for block_id in range(block_num):
                datas.append( utils.load_pickle( lgb_ans_dir+'{}_{}_{}_{}_0.005_ans.pkl'.format(v2versions[name], 'test', cur_stage, block_id ) ) )
            data = pd.concat( datas )
            get_result(ans=data,shift=0.0,bottom=0.7,after_deal=True,save_version='v2_'+name)
        '''
        #save2
        
        if ensemble_mode == 'ensemble':
            v1s = [
                ('big','0611052609'),
                ('fut1','0611052612'),
                ('fut4','0611052613'),
            ]
            v2s = [
                ('big2','0611052753'),
                ('fut12','0611052638'),
                ('fut42','0611052706')
            ]

            for i in range(len(v1s)):
                ver1 = v1s[i][0]
                ver2 = v2s[i][0]
                ans1 = data_read(v1s[i][1], sta='0.005')
                ans2 = data_read(v2s[i][1], sta='0.005')
                ans = ans1.copy()
                ans['label'] = ans1['label']*0.6 + ans2['label']*0.4
                print(f'v1: {ver1} , v2: {ver2}' )
                get_result(ans=ans,shift=0.0,bottom=0.7,after_deal=True,save_version=f'{ver1}-{ver2}-64merge-0.7_tmp')

            versions = [ ('big','big2'),('fut1','fut12'),('fut4','fut42') ]
            anss = {}
            for name in versions:
                data = pd.read_csv(prediction_result+f'{name[0]}-{name[1]}-64merge-0.7_tmp.csv',header=None)
                anss[name] = data

            df_test_stage = utils.load_pickle(online_all_test_data_path.format(cur_stage))
            tdata = df_test_stage.groupby('user_id').first()
            user2stage = dict( tdata['stage'] )
            users = anss[ ('big','big2') ][0].tolist()
            ans = []
            for i in range(len(users)):
                stage = user2stage[ users[i] ]
                if stage==9 :
                    ans.append( anss[ ('fut1','fut12') ].iloc[i] )
                elif stage==8 :
                    ans.append( anss[ ('fut4','fut42') ].iloc[i] )
                else:
                    ans.append( anss[ ('big','big2') ].iloc[i] )
            ans = pd.concat(ans,axis=1).T
            ans.to_csv(prediction_result+'result.csv',index=False,header=None)
        else:
            big_ver = '0611052609'
            ans = data_read(big_ver, sta='0.005')
            get_result(ans=ans,shift=0.0,bottom=0.7,after_deal=True,save_version='result')


        