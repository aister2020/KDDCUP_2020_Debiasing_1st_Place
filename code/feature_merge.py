#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from constants import *
import numpy as np
import pandas as pd
import utils


    
if __name__ == '__main__':
    seed = SEED
    cur_stage = CUR_STAGE
    neg = NEG
    mode = cur_mode
    #used_recall_source = 'i2i_w02-b2b-i2i2i'
    #used_recall_source = 'i2i_w02-b2b-i2i2i-i2i_w10'
    #used_recall_source = 'i2i_w02-b2b-i2i2i-i2i_w10-i2i2b'
    #used_recall_source = 'i2i_w02-b2b-i2i2i-i2i2b'
    used_recall_source = cur_used_recall_source
    sum_mode = 'nosum'
    used_recall_source = used_recall_source+'-'+sum_mode
    print( f'Recall Source Use {used_recall_source} Using mode {mode}')
    feat_names = [ 'feat_item_sum_sim_weight', 'feat_item_mean_sim_weight',
              'feat_item_sum_loc_weight', 'feat_item_mean_loc_weight',
              'feat_item_sum_time_weight', 'feat_item_mean_time_weight',
              'feat_item_sum_rank_weight', 'feat_item_mean_rank_weight',
              'feat_sum_sim_loc_time_weight',
              'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', #'feat_road_item_text_mansim',
             # 'feat_road_item_image_cossim', 'feat_road_item_image_eulasim', 'feat_road_item_image_mansim',
              'feat_sim_base', 'feat_sim_three_weight',
              'feat_different_type_road_score_sum_mean',
              #'feat_automl_cate_count','feat_automl_user_cate_count'
             ]
    feat_names = [ 'feat_item_sum_sim_weight', 'feat_item_mean_sim_weight',
              'feat_item_sum_loc_weight', 'feat_item_mean_loc_weight',
              'feat_item_sum_time_weight', 'feat_item_mean_time_weight',
              'feat_item_sum_rank_weight', 'feat_item_mean_rank_weight',
              'feat_sum_sim_loc_time_weight',
              'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', #'feat_road_item_text_mansim',
             # 'feat_road_item_image_cossim', 'feat_road_item_image_eulasim', 'feat_road_item_image_mansim',
              'feat_sim_base', 'feat_sim_three_weight',
              'feat_different_type_road_score_sum_mean',
              'feat_u2i_road_item_time_diff'
              #'feat_automl_cate_count','feat_automl_user_cate_count'
             ]    
    feat_names = [ 'feat_item_sum_sim_weight', 'feat_item_mean_sim_weight',
              'feat_item_sum_loc_weight', 'feat_item_mean_loc_weight',
              'feat_item_sum_time_weight', 'feat_item_mean_time_weight',
              'feat_item_sum_rank_weight', 'feat_item_mean_rank_weight',
              'feat_sum_sim_loc_time_weight',
              'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', #'feat_road_item_text_mansim',
             # 'feat_road_item_image_cossim', 'feat_road_item_image_eulasim', 'feat_road_item_image_mansim',
              'feat_sim_base', 'feat_sim_three_weight',
              'feat_different_type_road_score_sum_mean',
              'feat_u2i_road_item_time_diff',
              'feat_road_item_text_dot', 'feat_road_item_text_norm2',
              'feat_time_window_cate_count', #'feat_different_type_road_score_sum_mean_new'
              #'feat_automl_cate_count','feat_automl_user_cate_count'
             ]
    feat_names = [ 'feat_item_sum_sim_weight', 'feat_item_mean_sim_weight',
            'feat_item_sum_loc_weight', 'feat_item_mean_loc_weight',
            'feat_item_sum_time_weight', 'feat_item_mean_time_weight',
            'feat_item_sum_rank_weight', 'feat_item_mean_rank_weight',
            'feat_sum_sim_loc_time_weight',
            'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', #'feat_road_item_text_mansim',
            # 'feat_road_item_image_cossim', 'feat_road_item_image_eulasim', 'feat_road_item_image_mansim',
            'feat_sim_base', 'feat_sim_three_weight',
            'feat_different_type_road_score_sum_mean',
            'feat_u2i_road_item_time_diff',
            'feat_road_item_text_dot', 'feat_road_item_text_norm2',
            'feat_time_window_cate_count', #'feat_different_type_road_score_sum_mean_new'
            #'feat_automl_cate_count','feat_automl_user_cate_count'
            'feat_automl_recall_type_cate_count', 'feat_automl_loc_diff_cate_count', 'feat_automl_user_and_recall_type_cate_count',
            'feat_i2i_cijs_topk_by_loc', 'feat_i2i_cijs_median_mean_topk' ]

    
    feat_names = [ 'feat_item_sum_sim_weight', 'feat_item_mean_sim_weight',
        'feat_item_sum_loc_weight', 'feat_item_mean_loc_weight',
        'feat_item_sum_time_weight', 'feat_item_mean_time_weight',
        'feat_item_sum_rank_weight', 'feat_item_mean_rank_weight',
        'feat_sum_sim_loc_time_weight',
        'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', #'feat_road_item_text_mansim',
        # 'feat_road_item_image_cossim', 'feat_road_item_image_eulasim', 'feat_road_item_image_mansim',
        'feat_sim_base', 'feat_sim_three_weight',
        'feat_different_type_road_score_sum_mean',
        'feat_u2i_road_item_time_diff',
        'feat_road_item_text_dot', 'feat_road_item_text_norm2',
        'feat_time_window_cate_count', #'feat_different_type_road_score_sum_mean_new'
        #'feat_automl_cate_count','feat_automl_user_cate_count'
        'feat_automl_recall_type_cate_count', 'feat_automl_loc_diff_cate_count', 'feat_automl_user_and_recall_type_cate_count',
        'feat_i2i_cijs_topk_by_loc', 'feat_i2i_cijs_median_mean_topk',
        'item_recall_cnt_around_qtime' ]

    feat_names = [ 'feat_item_sum_sim_weight', 'feat_item_mean_sim_weight',
        'feat_item_sum_loc_weight', 'feat_item_mean_loc_weight',
        'feat_item_sum_time_weight', 'feat_item_mean_time_weight',
        'feat_item_sum_rank_weight', 'feat_item_mean_rank_weight',
        'feat_sum_sim_loc_time_weight',
        'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', #'feat_road_item_text_mansim',
        # 'feat_road_item_image_cossim', 'feat_road_item_image_eulasim', 'feat_road_item_image_mansim',
        'feat_sim_base', 'feat_sim_three_weight',
        'feat_different_type_road_score_sum_mean',
        'feat_u2i_road_item_time_diff',
        'feat_road_item_text_dot', 'feat_road_item_text_norm2',
        'feat_time_window_cate_count', #'feat_different_type_road_score_sum_mean_new'
        #'feat_automl_cate_count','feat_automl_user_cate_count'
        'feat_automl_recall_type_cate_count', 'feat_automl_loc_diff_cate_count', 'feat_automl_user_and_recall_type_cate_count',
        'feat_i2i_cijs_topk_by_loc', 'feat_i2i_cijs_median_mean_topk',
        'item_recall_cnt_around_qtime','feat_different_type_road_score_sum_mean_by_item' ]

    feat_names = [ 'feat_item_sum_sim_weight', 'feat_item_mean_sim_weight',
        'feat_item_sum_loc_weight', 'feat_item_mean_loc_weight',
        'feat_item_sum_time_weight', 'feat_item_mean_time_weight',
        'feat_item_sum_rank_weight', 'feat_item_mean_rank_weight',
        'feat_sum_sim_loc_time_weight',
        'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', #'feat_road_item_text_mansim',
        # 'feat_road_item_image_cossim', 'feat_road_item_image_eulasim', 'feat_road_item_image_mansim',
        'feat_sim_base', 'feat_sim_three_weight',
        'feat_different_type_road_score_sum_mean',
        'feat_u2i_road_item_time_diff',
        'feat_road_item_text_dot', 'feat_road_item_text_norm2',
        'feat_time_window_cate_count', #'feat_different_type_road_score_sum_mean_new'
        #'feat_automl_cate_count','feat_automl_user_cate_count'
        'feat_automl_recall_type_cate_count', 'feat_automl_loc_diff_cate_count', 'feat_automl_user_and_recall_type_cate_count',
        'feat_i2i_cijs_topk_by_loc', 'feat_i2i_cijs_median_mean_topk',
        'item_recall_cnt_around_qtime','feat_different_type_road_score_sum_mean_by_item','item_cnt_in_stage2' ]
    

    feat_names = [ 'feat_item_sum_sim_weight', 'feat_item_mean_sim_weight',
        'feat_item_sum_loc_weight', 'feat_item_mean_loc_weight',
        'feat_item_sum_time_weight', 'feat_item_mean_time_weight',
        'feat_item_sum_rank_weight', 'feat_item_mean_rank_weight',
        'feat_sum_sim_loc_time_weight',
        'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', #'feat_road_item_text_mansim',
        # 'feat_road_item_image_cossim', 'feat_road_item_image_eulasim', 'feat_road_item_image_mansim',
        'feat_sim_base', 'feat_sim_three_weight',
        'feat_different_type_road_score_sum_mean',
        'feat_u2i_road_item_time_diff',
        'feat_road_item_text_dot', 'feat_road_item_text_norm2',
        'feat_time_window_cate_count', #'feat_different_type_road_score_sum_mean_new'
        #'feat_automl_cate_count','feat_automl_user_cate_count'
        'feat_automl_recall_type_cate_count', 'feat_automl_loc_diff_cate_count', 'feat_automl_user_and_recall_type_cate_count',
        'feat_i2i_cijs_topk_by_loc', 'feat_i2i_cijs_median_mean_topk',
        'item_recall_cnt_around_qtime','feat_different_type_road_score_sum_mean_by_item','item_cnt_in_stage2',
        'feat_item_cnt_in_different_stage', 
        'feat_different_type_road_score_mean_by_road_item',
        'feat_i2i2i_sim',
        'feat_numerical_groupby_item_cnt_in_stage'
        #'feat_base_info_in_stage', 
        #'feat_item_time_info_in_stage',, 'feat_user_info_in_stage'
        #'feat_user_and_item_count_in_three_init_data'
        ]



    feat_names = [ 'feat_item_sum_mean_sim_weight_loc_weight_time_weight_rank_weight',
        'feat_sum_sim_loc_time_weight',
        'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', 
        'feat_sim_base', 'feat_sim_three_weight',
        'feat_different_type_road_score_sum_mean', 'feat_u2i_road_item_time_diff',
        'feat_road_item_text_dot', 'feat_road_item_text_norm2',
        'feat_time_window_cate_count',
        'feat_automl_recall_type_cate_count', 'feat_automl_loc_diff_cate_count', 
        'feat_automl_user_and_recall_type_cate_count', 'feat_i2i_cijs_topk_by_loc', 'feat_i2i_cijs_median_mean_topk',
        'feat_different_type_road_score_sum_mean_by_item','item_cnt_in_stage2',
        'feat_item_cnt_in_different_stage', 'feat_different_type_road_score_mean_by_road_item',
        'feat_i2i2i_sim',
        'feat_item_stage_nunique', 'feat_item_qtime_time_diff', 'feat_item_cumcount', 'feat_road_time_bins_cate_cnt',
        'feat_u2i_road_item_before_and_after_query_time_diff',
        'feat_item_cnt_in_stage2_mean_max_min_by_user',
        'feat_item_seq_sim_cossim_text',
        'feat_item_seq_sim_cossim_image',
        'feat_i2i_sim_on_hist_seq',
        'feat_item_max_sim_weight_loc_weight_time_weight_rank_weight', 
        'feat_different_type_road_score_max',
        'feat_different_type_road_score_max_by_item', 
        'feat_item_sum_mean_max_i2i2i_weight',
        'feat_item_sum_mean_max_sim_weight_loc_weight_time_weight_rank_weight_by_item',
        'feat_item_sum_mean_max_i2i2i_weight_by_item'
        ]



    feat_names = [ 'feat_item_sum_mean_sim_weight_loc_weight_time_weight_rank_weight',
        'feat_sum_sim_loc_time_weight',
        'feat_road_item_text_cossim', 'feat_road_item_text_eulasim', 
        'feat_sim_base', 'feat_sim_three_weight',
        'feat_different_type_road_score_sum_mean', 'feat_u2i_road_item_time_diff',
        'feat_road_item_text_dot', 'feat_road_item_text_norm2',
        'feat_time_window_cate_count',
        'feat_automl_recall_type_cate_count', 'feat_automl_loc_diff_cate_count', 
        'feat_automl_user_and_recall_type_cate_count', 'feat_i2i_cijs_topk_by_loc', 'feat_i2i_cijs_median_mean_topk',
        'feat_different_type_road_score_sum_mean_by_item','item_cnt_in_stage2',
        'feat_item_cnt_in_different_stage', 'feat_different_type_road_score_mean_by_road_item',
        'feat_i2i2i_sim',
        'feat_item_stage_nunique', 'feat_item_qtime_time_diff', 'feat_item_cumcount', 'feat_road_time_bins_cate_cnt',
        'feat_u2i_road_item_before_and_after_query_time_diff',
        'feat_item_cnt_in_stage2_mean_max_min_by_user',
        'feat_item_seq_sim_cossim_text',
        'feat_item_seq_sim_cossim_image',
        'feat_i2i_sim_on_hist_seq',
        ]

    print('feature_merge num: ', len(feat_names))
    base = utils.load_pickle( lgb_base_pkl.format( used_recall_source,mode,cur_stage) )    
    
    feat_paths = [ (i+'_{}_{}.pkl').format(mode,cur_stage) for i in feat_names ]
    
    feat_list = [base]

    for feat_path in feat_paths:
        feat = utils.load_pickle( feat_dir + f'{used_recall_source}/' + feat_path )
        print(feat_path, ':',feat.shape)
        for col in feat.columns:
            feat[col] = utils.downcast(feat[col])
        feat_list.append( feat )
    data = pd.concat( feat_list, axis=1 ).reset_index(drop=True)
    print('[INFO] after merge feature, data shape is ', data.shape)
    print('[INFO] features: ', data.columns)
    block_len = len(data)//merge_block_num
    for block_id in range(merge_block_num):
        l = block_id * block_len
        r = (block_id+1) * block_len
        print('merging block: ', block_id)
        if block_id == merge_block_num - 1:
            utils.dump_pickle( data.iloc[l:], lgb_model_pkl.format( used_recall_source,mode,cur_stage,block_id ) )
        else:
            utils.dump_pickle( data.iloc[l:r], lgb_model_pkl.format( used_recall_source,mode,cur_stage,block_id ) )
    '''
    datas = []
    for block_id in range(block_num):
        datas.append( utils.load_pickle( lgb_model_pkl.format( used_recall_source,mode,cur_stage,block_id ) ) )
    data = pd.concat( datas )
    '''