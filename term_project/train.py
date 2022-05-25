import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

# pip install xgboost
import xgboost
from xgboost import XGBClassifier

# pip install catboost
# pip install category_encoders
from catboost import CatBoostClassifier, Pool

####################
# Load training data
####################

feature_names = ['age', 'sex', 'education', 'marital status', 'card limit']
feature_names.extend(['use_' + str(i) for i in range(1, 7)])
feature_names.extend(['pay_' + str(i) for i in range(1, 7)])
df_input_base = pd.read_table('./_datasets/train_input.txt', delimiter=',', header=None, names=feature_names)
print(df_input_base)

####################
# 1. 주어진 데이터를 이해하기 위한 각종 분석
####################

# 1) Feature별 데이터 분포 확인
fig, axs = plt.subplots(5, 4)
plt.subplots_adjust(hspace=0.5, wspace=0.3)
fig.set_figheight(12)
fig.set_figwidth(12)
categorical_feature_names = ['sex', 'education', 'marital status']
for index, feature_name in enumerate(feature_names):
    i_row = int(np.floor((index / 4) % 5))
    i_col = index % 4
    axs[i_row, i_col].set_title(feature_name)
    axs[i_row, i_col].hist(df_input_base[feature_name], alpha=0.4, align='mid', color='green')

    if feature_name in categorical_feature_names:
        unique_values = sorted(df_input_base[feature_name].unique())
        axs[i_row, i_col].set_xticks(unique_values)
plt.savefig('./results/data_distribution.png')


'''
데이터 마이닝 기법을 이용한 신용카드 연체고객 예측 및 감소 방안 제시

데이터는 인적 특성 변수 9개(나이, 성별, 거주지역, 우편번호, 연간 소
득, 직업유형 등)와 행태특성 변수 73개(이용한도, 연간 이용 회수, 연간
이용금액, 연체회차, 연체금액, 사이버 회원 여부, SMS 사용 여부, 명세서
발송 매체 등) 총 82개의 항목으로 구성되어 있다.

https://brunch.co.kr/@tobesoft-ai/15
https://dacon.io/competitions/official/235713/codeshare/2768?page=1&dtype=recent
'''

####################
# Feature extraction
####################

####################
# 2. 데이터에 대한 적절한 처리 및 특징 추출 방안
####################

# 2) Permutation feature importance : https://sarah0518.tistory.com/53

# 1) Feature 값 치환
df_input_pp = df_input_base.copy()
# 최종학력 : 1,2,3,4 이외 값은 4(기타)로 변경
df_input_pp['education'] = df_input_pp['education'].map(lambda x: 4 if x not in [1, 2, 3, 4] else x)
# 결혼여부 : 1,2,3 이외 값은 3(기타)로 변경
df_input_pp['marital status'] = df_input_pp['marital status'].map(lambda x: 3 if x not in [1, 2, 3] else x)

# 2) Feature 추가 : 과거 6개월간 청구대금/납부금액 → 이용금액(연단위 환산), 연체회차, 연체금액


# 3. 데이터셋 활용 방안(train data 적음, balance 맞지 않음)
# 4. 불균형이 심한 데이터를 처리 및 학습하기 위한 방안
# TODO oversampling



####################
# Model Training
####################

# 5. 2종 이상의 모델 설꼐 및 성능 비교

# 6. 성능을 향상시키기 위한 각종 아이디어

# MLP
# num_layers, num_neurons, num_epoch = 8, 128, 1000
# for epoch in range(100, num_epoch, 100):
#     mlp_model = MLPClassifier(hidden_layer_sizes=(num_layers, num_neurons),
#                               activation='relu',
#                               solver='adam',
#                               early_stopping=True,
#                               max_iter=epoch)
#     mlp_model = mlp_model.fit(train_input, train_target)
#     pred_target = mlp_model.predict(valid_input)
#     # balanced accuracy :  각 클래스 재현율(recall)의 산술 평균, [0,1]
#     b_accr = balanced_accuracy_score(valid_target, pred_target)
#     print('epoch: {},\tbalanced accuracy: {}'.format(epoch, b_accr))


# SVM
# svc_model = SVC(kernel='rbf')
# svc_model.fit(train_input, train_target)
# pred_target = svc_model.predict(valid_input)
# b_accr = balanced_accuracy_score(valid_target, pred_target)
# print('balanced accuracy: {}'.format(b_accr))

# XG boosting
# xgb_model = XGBClassifier(booster='gbtree', max_depth=10, gamma=0.5, learning_rate=0.01, n_estimators=100, random_state=99)
# xgb_model = xgb_model.fit(train_input, train_target)
# pred_target = xgb_model.predict(valid_input)
# b_accr = balanced_accuracy_score(valid_target, pred_target)
# print('balanced accuracy: {}'.format(b_accr))

# CatBoost
# cat_model = CatBoostClassifier()
# cat_model.fit(train_input, train_target, use_best_model=True, early_stopping_rounds=100, verbose=100)
# pred_target = cat_model.predict(valid_input)
# b_accr = balanced_accuracy_score(valid_target, pred_target)
# print('balanced accuracy: {}'.format(b_accr))

# FIXME Idea
# Ensemble : XGBoost, LightGBM, CatBoost
# 파라미터 최적화 : Scikit-learn GridSearchCV

####################
# Save model
####################
