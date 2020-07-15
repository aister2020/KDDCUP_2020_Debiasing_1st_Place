#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import *
import numpy as np
import pandas as pd
import utils
item2vecs = {}
with open(item_feat_csv) as fin:
    for line in fin:
        line = line.strip('\n')
        index = line.index(',')
        item_id = int(line[:index])
        index2 = line.index(']')
        vec1 = np.array(eval(line[index+1:index2+1]))
        vec2 = np.array(eval(line[index2+2:]))
        item2vecs[item_id] = [vec1,vec2]
        
utils.dump_pickle(item2vecs,item_feat_pkl)

df_user = pd.read_csv(user_feat_csv,names=['user_id','user_age_level','user_gender','user_city_level'])
df_user['user_id'] = df_user['user_id'].astype(np.int32)
df_user['user_age_level'] = df_user['user_age_level'].fillna(0).astype(np.int32)

df_user['user_gender'] = df_user['user_gender'].map({'M':1,'F':2}).fillna(0).astype(np.int32)

df_user['user_city_level'] =  df_user['user_city_level'].fillna(0).astype(np.int32)

utils.dump_pickle(df_user,user_feat_pkl)

