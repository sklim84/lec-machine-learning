# Credit: Hojung Lee (hjlee92@yonsei.ac.kr)
# Usage limited to the students of the graduate course "Machine Learning and Pattern Recognition" in 2022 at Department of Artificial Intelligence, Yonsei University

import os, sys, json, pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TV_path = ['Training/Training_라벨링데이터', 'Validation/[V라벨링]라벨링데이터']
feature = ["DAMAGE", "TRANSPARENCY", "Shape", "Texture", "Object Size"]

damage = ['완전훼손', '상당훼손', '일부훼손']
transparency = ['불투명', '투명']
shape = ['다면체', '원통형', '원형', '직육면체', '평면형']
texture = ['딱딱함', '부드러움', '혼합']
size = ['대', '중', '소']

NUM_TRAIN_DATA_EACH_CLASS = 3000
NUM_TEST_DATA_EACH_CLASS = 387
NUM_CLASSES = 15

############# Train Data Pre-processing #############
d = TV_path[0]
a = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

label_path = [] # 15개
for i in a:
    b = [os.path.join(i, o) for o in os.listdir(i) if os.path.isdir(os.path.join(i,o))]
    for j in b:
        c = [os.path.join(j, o) for o in os.listdir(j) if os.path.isdir(os.path.join(j,o))]
        if len(c) > NUM_TRAIN_DATA_EACH_CLASS: # 15개
            label_path.append(j)

train_data = []
for i in label_path: # class
    d = [o for o in os.listdir(i) if os.path.isdir(os.path.join(i,o))]
    class_data = []
    for j in d: # each file
        if len(class_data) == NUM_TRAIN_DATA_EACH_CLASS:
            break
        try:
            with open(i + '/' + j + '/' + j + '_0.Json', encoding='utf-8') as f:
                data = json.load(f)
                data_ = []
                for k in feature:
                    data_.append(data['Bounding'][0][k])
                class_data.append(data_)
        except:
            continue
    train_data.append(class_data)
train_data = np.array(train_data)

###### normalize
train_data[train_data == damage[0]] = 0/2
train_data[train_data == damage[1]] = 1/2
train_data[train_data == damage[2]] = 2/2

train_data[train_data == transparency[0]] = 0
train_data[train_data == transparency[1]] = 1

train_data[train_data == shape[0]] = 0/4
train_data[train_data == shape[1]] = 1/4
train_data[train_data == shape[2]] = 2/4
train_data[train_data == shape[3]] = 3/4
train_data[train_data == shape[4]] = 4/4

train_data[train_data == texture[0]] = 0/2
train_data[train_data == texture[1]] = 1/2
train_data[train_data == texture[2]] = 2/2

train_data[train_data == size[0]] = 0/2
train_data[train_data == size[1]] = 1/2
train_data[train_data == size[2]] = 2/2

train_data = np.reshape(train_data, (NUM_CLASSES * NUM_TRAIN_DATA_EACH_CLASS, 5)) # (15, 3000, 5)

#train_label = np.arange(1, 16)
#train_label = np.repeat(train_label, NUM_TRAIN_DATA_EACH_CLASS, axis=0)

with open('train.pkl','wb') as f:
    pickle.dump(train_data, f)

############# Test Data Pre-processing #############
test_label_path = []
for i in label_path:
    i = i.replace(TV_path[0], TV_path[1])
    test_label_path.append(i)

test_data = []
for i in test_label_path: # class
    d = [o for o in os.listdir(i) if os.path.isdir(os.path.join(i,o))]
    class_data = []
    for j in d: # each file
        if len(class_data) == NUM_TEST_DATA_EACH_CLASS:
            break
        try:
            with open(i + '/' + j + '/' + j + '_0.Json', encoding='utf-8') as f:
                data = json.load(f)
                data_ = []
                for k in feature:
                    data_.append(data['Bounding'][0][k])
                class_data.append(data_)
        except:
            continue
    test_data.append(class_data)
test_data = np.array(test_data)

###### normalize
test_data[test_data == damage[0]] = 0/2
test_data[test_data == damage[1]] = 1/2
test_data[test_data == damage[2]] = 2/2

test_data[test_data == transparency[0]] = 0
test_data[test_data == transparency[1]] = 1

test_data[test_data == shape[0]] = 0/4
test_data[test_data == shape[1]] = 1/4
test_data[test_data == shape[2]] = 2/4
test_data[test_data == shape[3]] = 3/4
test_data[test_data == shape[4]] = 4/4

test_data[test_data == texture[0]] = 0/2
test_data[test_data == texture[1]] = 1/2
test_data[test_data == texture[2]] = 2/2

test_data[test_data == size[0]] = 0/2
test_data[test_data == size[1]] = 1/2
test_data[test_data == size[2]] = 2/2

test_data = np.reshape(test_data, (NUM_CLASSES * NUM_TEST_DATA_EACH_CLASS, 5))

#test_label = np.arange(1, 16)
#test_label = np.repeat(test_label, NUM_TEST_DATA_EACH_CLASS, axis=0)

with open('test.pkl','wb') as f:
    pickle.dump(test_data, f)
