# -*- coding: utf-8 -*-

import numpy as np
from constants import *
import utils
import gc

item_feat = utils.load_pickle(item_feat_pkl)
feat_item_set = set(item_feat.keys())

item_vec1= np.zeros((120000,128),dtype='float32')
item_vec2= np.zeros((120000,128),dtype='float32')

for k,v in item_feat.items():
    item_vec1[k] = v[0]
    item_vec2[k] = v[1]


split_size = 1000
split_num = int(item_vec1.shape[0]/split_size)
if item_vec1.shape[0]%split_size != 0:
    split_num += 1

all_idx = []
all_score = []

l2norm1 = np.linalg.norm(item_vec1,axis=1,keepdims=True)
item_vec1 = item_vec1/(l2norm1+1e-9)

l2norm2 = np.linalg.norm(item_vec2,axis=1,keepdims=True)
item_vec2 = item_vec2/(l2norm2+1e-9)

vec1_vec2 = np.transpose(item_vec1)

vec2_vec2 = np.transpose(item_vec2)

for i in range(split_num):
    vec1_vec1 = item_vec1[i*split_size:(i+1)*split_size]
    vec2_vec1 = item_vec2[i*split_size:(i+1)*split_size]
    
    text_sim = vec1_vec1.dot(vec1_vec2)
    image_sim = vec2_vec1.dot(vec2_vec2)
    
    blend_sim = 0.95*text_sim + 0.05*image_sim
    
    idx = (-blend_sim).argsort(axis=1)
    blend_sim = (-blend_sim)
    blend_sim.sort(axis=1)
    idx = idx[:,:500]
    score = blend_sim[:,:500]
    score = -score
    all_idx.append(idx)
    all_score.append(score)
    
    gc.collect()
    print('split_num',i)
    

idx = np.concatenate(all_idx)    
score = np.concatenate(all_score)

sim = []
for i in range(idx.shape[0]):
    if i in feat_item_set:
        sim_i = []
        for j,item in enumerate(idx[i]):
            if item in feat_item_set:
                sim_i.append((item,score[i][j]))

        sim.append((i,sim_i))



utils.write_sim(sim,item_blend_sim_path)