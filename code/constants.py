# -*- coding: utf-8 -*-

import os 


####路径
data_dir = '../data/'

cur_mode = 'test'
cur_used_recall_source = 'i2i_w02-b2b-i2i2i-i2i_w10-i2i2b-b2b2b-b2bl2'
#cur_used_recall_source = 'i2i_w02-b2b-i2i2i-i2i_w10-i2i2b'
#cur_used_recall_source = 'i2i_w02-b2b-i2i2i-i2i_w10-i2i2b-b2bl2'
cur_used_recall_source = 'i2i_w10-b2b-i2i2i_new'
#cur_used_recall_source = 'i2i_w02-b2b-i2i2i_new'
item_feat_csv = data_dir+'underexpose_train/underexpose_item_feat.csv'
user_feat_csv = data_dir+'underexpose_train/underexpose_user_feat.csv'

train_item_click_csv = data_dir+'underexpose_train/underexpose_train_click-{}.csv'
def get_train_item_click_file(stage):
    return train_item_click_csv.format(stage)

test_item_click_csv = data_dir+'underexpose_test/underexpose_test_click-{}/underexpose_test_click-{}.csv'
def get_test_item_click_file(stage):
    return test_item_click_csv.format(stage,stage)

test_qtime_csv = data_dir+'underexpose_test/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'
def get_test_qitem_file(stage):
    return test_qtime_csv.format(stage,stage)

prediction_result = '../prediction_result/'
if not os.path.exists(prediction_result):
    os.makedirs(prediction_result)

pkl_dir = '../user_data/pkl_data/'
if not os.path.exists(pkl_dir):
    os.makedirs(pkl_dir)
    
item_feat_pkl = pkl_dir+"item_feat.pkl"
user_feat_pkl = pkl_dir+"user_feat.pkl"

item2time_path = pkl_dir+"item2time_{}_{}.pkl"
stage_occur_items_path = pkl_dir+"stage_occur_items_{}_{}.pkl"
item_pair2time_diff_path = pkl_dir+"item_pair2times_{}_{}.pkl"
item_pair2time_seq_path = pkl_dir+"item_pair2time_seq_{}_{}.pkl"
item2times_path = pkl_dir+"item2times_{}_{}_{}.pkl"

rerank_path = pkl_dir+'rerank_from_{}_{}.pkl'

lgb_df_pkl = '../user_data/pkl_data/lgb_df_{}.pkl'
#lgb_processed_df_pkl = '../user_data/pkl_data/lgb_processed_df_{}.pkl'
lgb_base_pkl = '../user_data/pkl_data/{}/lgb_base_{}_{}.pkl'
block_num = 6
merge_block_num = 12
lgb_model_pkl = '../user_data/pkl_data/{}/lgb_model_{}_{}_{}.pkl'

lgb_model_dir = '../user_data/lgb_model/'
if not os.path.exists(lgb_model_dir):
    os.makedirs(lgb_model_dir)

lgb_ans_dir = '../user_data/lgb_ans/'
if not os.path.exists(lgb_ans_dir):
    os.makedirs(lgb_ans_dir)

lgb_pivot_dir = '../user_data/lgb_pivot/'
if not os.path.exists(lgb_pivot_dir):
    os.makedirs(lgb_pivot_dir)

lgb_ensemble_dir = '../user_data/lgb_ensemble/'
if not os.path.exists(lgb_ensemble_dir):
    os.makedirs(lgb_ensemble_dir)

feat_dir = '../user_data/feat_data/'
if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

feat_imp_dir = '../user_data/feat_imp/'
if not os.path.exists(feat_imp_dir):
    os.makedirs(feat_imp_dir)
    
training_dir = '../user_data/training_data/'
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

default_neg_sample_path = 'default_neg_sample'
default_neg_tfrecord_path = 'default_neg_tfrecord'
default_cfrank_neg_tfrecord_path = 'default_cfrank_neg_tfrecord_path'

train_with_neg_pkl = 'train_with_neg_{}.pkl'
# test_train_with_neg_pkl = 'test_train_with_neg_{}.pkl'
valid_with_neg_pkl = 'valid_with_neg_{}.pkl'

train_tfrecord = 'train_{}.tfrecord'
valid_tfrecord = 'valid_{}.tfrecord'
cv_train_tfrecord = 'cv_train_{}_{}.tfrecord'
cv_valid_tfrecord = 'cv_valid_{}_{}.tfrecord'
i2i_item_tfrecord = 'i2i_item_{}.tfrecord'

predict_dir = '../user_data/predict_data/'

if not os.path.exists(predict_dir):
    os.makedirs(predict_dir)

default_predict_valid_tfrecord = training_dir+'predict_valid_{}.tfrecord'
default_predict_valid_answer = training_dir+'predict_valid_answer_{}.pkl'

default_predict_valid_part_tfrecord = training_dir+'predict_valid_part_{}.tfrecord'
default_predict_valid_part_answer = training_dir+'predict_valid_part_answer_{}.pkl'

default_predict_test_tfrecord = training_dir+'predict_test_{}.tfrecord'
default_predict_test_index = training_dir+'predict_test_index_{}.pkl'

all_train_data_path = pkl_dir+'all_train_data_{}.pkl'
all_train_stage_data_path = pkl_dir+'all_train_stage_data_{}.pkl'
all_valid_data_path = pkl_dir+'all_valid_data_{}.pkl'
all_valid_stage_data_path = pkl_dir+'all_valid_stage_data_{}.pkl'
all_test_data_path = pkl_dir+'all_test_data_{}.pkl'

online_all_train_data_path = pkl_dir+'online_all_train_data_{}.pkl'
online_all_train_stage_data_path = pkl_dir+'online_all_train_stage_data_{}.pkl'
online_all_test_data_path = pkl_dir+'online_all_test_data_{}.pkl'

full_item_degree_path = pkl_dir+'full_item_degree_{}.pkl'
phase_full_item_degree_path = pkl_dir+'phase_full_item_degree_{}.pkl'

user2recall_path = pkl_dir+'user2recall_{}.pkl'
user2recall_textsim_path = pkl_dir+'user2recall_textsim_{}.pkl'
user2recall_imagesim_path = pkl_dir+'user2recall_imagesim_{}.pkl'

recall_path = pkl_dir+'recall_{}_{}_{}.pkl'

answer_source_path = pkl_dir+'answer_source_{}_{}.pkl'
i2i_w02_recall_scoure_path = pkl_dir+'i2i_w02_recall_source_{}_{}_{}.pkl'
b2b_recall_scoure_path = pkl_dir+'b2b_recall_source_{}_{}_{}.pkl'
i2i2i_recall_scoure_path = pkl_dir+'i2i2i_recall_source_{}_{}_{}.pkl'
i2i2i_new_recall_scoure_path = pkl_dir+'i2i2i_new_recall_source_{}_{}_{}.pkl'
i2i_w10_recall_scoure_path = pkl_dir+'i2i_w10_recall_source_{}_{}_{}.pkl'
b2b2b_recall_scoure_path = pkl_dir+'b2b2b_recall_source_{}_{}_{}.pkl'
i2i2b_recall_scoure_path = pkl_dir+'i2i2b_recall_source_{}_{}_{}.pkl'
b2b2i_recall_scoure_path = pkl_dir+'b2b2i_recall_source_{}_{}_{}.pkl'
b2bl2_recall_scoure_path = pkl_dir+'b2bl2_recall_source_{}_{}_{}.pkl'

answer_itemrecall_source_path = pkl_dir+'answer_itemrecall_source_{}_{}_{}.pkl'
itemrecall_source_path = pkl_dir+'itemrecall_source_{}_{}_{}.pkl'
textrecall_source_path = pkl_dir+'textrecall_source_{}_{}_{}.pkl'


online_train_recall_path = pkl_dir+'online_train_recall_{}.pkl'
online_test_recall_path = pkl_dir+'online_train_recall_{}.pkl'

predict_stage_path = pkl_dir+'predict_stage_{}.pkl'
online_predict_stage_path = pkl_dir+'online_predict_stage_{}.pkl'

item_sim_path = pkl_dir+'item_sim_{}_{}.pkl'
item_text_sim_path = pkl_dir+'item_text_sim.text'
item_text_l2_sim_path = pkl_dir+'item_text_l2_sim.text'

item_image_sim_path = pkl_dir+'item_image_sim.text'
item_blend_sim_path = pkl_dir+'item_blend_sim.text'

log_dir = '../user_data/log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

####常量
SEED = 2020
NEG = 5

PREDICT_ITEM_NUM = 50

CUR_STAGE = 9
