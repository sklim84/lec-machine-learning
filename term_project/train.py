import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
# pip install xgboost
import xgboost
from xgboost import XGBClassifier

# pip install catboost
# pip install category_encoders
from catboost import CatBoostClassifier, Pool
from enum import Enum
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN
from collections import Counter


# 실행모드
class EXEMODE(Enum):
    ALL = 0
    FINAL = 1


# 데이터 분포 확인
def data_distribution(data):
    # 정상납부 고객 정보
    data_normal = data[data['target'] == 0].drop(['target'], axis=1)
    # 연체 고객 정보
    data_overdue = data[data['target'] == 1].drop(['target'], axis=1)
    # 1) Feature별 데이터 분포 확인
    fig, axs = plt.subplots(5, 4)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    for index, feature_name in enumerate(feature_names):
        i_row = int(np.floor((index / 4) % 5))
        i_col = index % 4
        axs[i_row, i_col].set_title(feature_name)
        axs[i_row, i_col].hist(data_normal[feature_name], alpha=0.4, align='mid', color='green', label='target=0')
        axs[i_row, i_col].hist(data_overdue[feature_name], alpha=0.4, align='mid', color='blue', label='target=1')
        axs[i_row, i_col].legend()

        if feature_name in feature_names_categorical:
            unique_values = sorted(data_base[feature_name].unique())
            axs[i_row, i_col].set_xticks(unique_values)
    plt.savefig('./results/data_distribution.png')

# 데이터 전처리
def data_preprocessing(data):
    print('##### number of data before preprocessing: {}'.format(data.shape[0]))
    result = []
    data_pp = data.copy()

    # 최종학력 : 1,2,3,4 이외 값 데이터 삭제
    # data_pp['education'] = data_pp['education'].map(lambda x: 4 if x not in [1, 2, 3, 4] else x)
    fault_edu = data_pp[~data_pp['education'].isin([1, 2, 3, 4])].index
    print('\tremove education data: {} rows'.format(len(fault_edu)))
    data_pp.drop(fault_edu, inplace=True)
    data_pp.reset_index(drop=True, inplace=True)
    result.append(('education', len(fault_edu)))

    # 결혼여부 : 1,2,3 이외 값 데이터 삭제
    # data_pp['marital_status'] = data_pp['marital_status'].map(lambda x: 3 if x not in [1, 2, 3] else x)
    fault_ms = data_pp[~data_pp['marital_status'].isin([1, 2, 3])].index
    print('\tremove marital status data: {} rows'.format(len(fault_ms)))
    data_pp.drop(fault_ms, inplace=True)
    data_pp.reset_index(drop=True, inplace=True)
    result.append(('marital_status', len(fault_ms)))

    # 과거 6개월간 월별 청구 대금 : 음수(-) 또는 신용카드 한도액 초과 시 데이터 삭제
    for feature_name_use in feature_names_use:
        fault_use = data_pp[(data_pp[feature_name_use] < 0) | (data_pp[feature_name_use] > data_pp['card_limit'])].index
        if len(fault_use) != 0:
            print('\tremove {} data: {} rows'.format(feature_name_use, len(fault_use)))
            data_pp.drop(fault_use, inplace=True)
            data_pp.reset_index(drop=True, inplace=True)
            result.append((feature_name_use, len(fault_use)))
    # data_pp[feature_names_use] = abs(data_pp[feature_names_use])
    # comp_use_limit = np.expand_dims(data_pp['card_limit'].to_numpy(), axis=1)
    # comp_use_limit = np.tile(comp_use_limit, reps=[1, len(feature_names_use)])  # 6개 열로 복사
    # comp_use_limit = comp_use_limit - data_pp[feature_names_use].to_numpy()
    # data_pp.drop(np.where(comp_use_limit < 0)[0], inplace=True)
    # data_pp.reset_index(drop=True, inplace=True)

    # 과거 6개월간 월별 납부 금액 : 음수(-) 데이터 삭제
    # data_pp[feature_names_pay] = abs(data_pp[feature_names_pay])
    for feature_name_pay in feature_names_pay:
        fault_pay = data_pp[data_pp[feature_name_pay] < 0].index
        if len(fault_pay) != 0:
            print('\tremove {} data: {} rows'.format(feature_name_pay, len(fault_pay)))
            data_pp.drop(fault_pay, inplace=True)
            data_pp.reset_index(drop=True, inplace=True)
            result.append((feature_name_pay, len(fault_pay)))

    result = pd.DataFrame(result, columns=['feature', 'number of removed data'])
    result.to_csv('./results/data_preprocessing.csv')
    print('##### number of data after preprocessing: {}'.format(data_pp.shape[0]))

    return data_pp

# feature 중요도 계산
def feature_importance(model, x, y, feature_names, postfix=None):
    if postfix == None:
        postfix = model.__class__.__name__

    # permutation importance
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.33, random_state=42)
    model.fit(train_x, train_y)
    print('complete model fit...')
    result = permutation_importance(model, valid_x, valid_y, n_repeats=30, random_state=0, n_jobs=3)
    sorted_idx = result.importances_mean.argsort()
    print('complete permutation importance...')

    # save csv
    importances = pd.DataFrame()
    importances['feature'] = [feature_names[index] for index in sorted_idx]
    importances['importance'] = result.importances_mean[sorted_idx]
    importances.sort_values(by='importance', ascending=False, inplace=True)
    importances.to_csv('./results/feature_importance_{}.csv'.format(postfix))
    print(importances)

    # save plot
    sns.set(rc={"axes.unicode_minus": False}, style='whitegrid')
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=[feature_names[index] for index in sorted_idx])
    ax.set_title("Feature Importance ({})".format(postfix))
    fig.tight_layout()
    plt.savefig('./results/feature_inportance_{}.png'.format(postfix))


# 실행모드
exe_mode = EXEMODE.FINAL

####################
# 1. 주어진 데이터를 이해하기 위한 각종 분석
####################
feature_names = ['age', 'sex', 'education', 'marital_status', 'card_limit']
feature_names_use = ['use_' + str(i) for i in range(1, 7)]
feature_names_pay = ['pay_' + str(i) for i in range(1, 7)]
feature_names.extend(feature_names_use)
feature_names.extend(feature_names_pay)
feature_names_categorical = ['sex', 'education', 'marital_status']
feature_names_numerical = ['age', 'card_limit']
feature_names_numerical.extend(feature_names_use)
feature_names_numerical.extend(feature_names_pay)
data_base = pd.read_table('./_datasets/train_input.txt', delimiter=',', header=None, names=feature_names)
data_base['target'] = np.loadtxt('./_datasets/train_target.txt', dtype=int)

# 1) 데이터 분포 확인
if exe_mode == EXEMODE.ALL:
    data_distribution(data_base)

####################
# 2. 데이터에 대한 적절한 처리 및 특징 추출 방안
####################

# 1) 데이터 전처리
data_pp = data_preprocessing(data_base)

# 2) Feature 중요도 평가
if exe_mode == EXEMODE.ALL:
    # SVM
    svc_model = SVC(kernel='rbf')
    input = data_pp.drop(['target'], axis=1).to_numpy()
    target = data_pp['target'].to_numpy()
    feature_importance(svc_model, input, target, feature_names)

    # XGBoost
    xgb_model = XGBClassifier(booster='gbtree', max_depth=10, gamma=0.5, learning_rate=0.01, n_estimators=100,
                              random_state=99)
    feature_importance(xgb_model, input, target, feature_names)

    # CatBoost
    cat_model = CatBoostClassifier()
    feature_importance(cat_model, input, target, feature_names)

# 3) Feature selection : 삭제(성별, 최종학력, 결혼여부), 추가(이용금액, 납부금액, 연체회차, 연체금액), 삭제(과거 6개월간 청구대금, 납부금액)
if exe_mode == EXEMODE.ALL:
    data_fs = data_pp.copy()

    data_fs.drop(['sex'], axis=1, inplace=True)
    data_fs.drop(['education'], axis=1, inplace=True)
    data_fs.drop(['marital_status'], axis=1, inplace=True)

    np_use = data_fs[feature_names_use].to_numpy()
    np_pay = data_fs[feature_names_pay].to_numpy()
    # 연체회차 추가
    np_overdue = np_use - np_pay
    np_overdue[np_overdue > 0] = 1
    np_overdue[np_overdue <= 0] = 0
    data_fs['overdue_num'] = np.sum(np_overdue, axis=1)
    feature_names.extend(['overdue_num'])
    feature_names_numerical.extend(['overdue_num'])
    # 납부금액 추가
    np_pay = data_fs[feature_names_pay].to_numpy()
    data_fs['pay_amt'] = np.sum(np_pay, axis=1)
    feature_names.extend(['pay_amt'])
    feature_names_numerical.extend(['pay_amt'])
    # 이용금액 추가
    data_fs['use_amt'] = np.sum(np_use, axis=1)
    feature_names.extend(['use_amt'])
    feature_names_numerical.extend(['use_amt'])
    # 연체금액 추가
    np_overdue_amt = np.sum(np_use - np_pay, axis=1)
    np_overdue_amt[np_overdue_amt > 0] = 0
    data_fs['overdue_amt'] = abs(np_overdue_amt)
    feature_names.extend(['overdue_amt'])
    feature_names_numerical.extend(['overdue_amt'])

    # 과거 6개월간 청구대금/납부금액 삭제
    data_fs.drop(feature_names_use, axis=1, inplace=True)
    data_fs.drop(feature_names_pay, axis=1, inplace=True)
    feature_names_numerical = [ele for ele in feature_names_numerical if ele not in feature_names_use]
    feature_names_numerical = [ele for ele in feature_names_numerical if ele not in feature_names_pay]

# 4) Feature scaling
# Scale에 따라 주성분의 설명 가능한 분산량이 왜곡될 수 있기 때문에 PCA 수행 전 표준화 필요
scaler = StandardScaler()
data_pp[feature_names_numerical] = scaler.fit_transform(data_pp[feature_names_numerical])

# 5) Feature extraction
# PCA (visualize : https://plotly.com/python/pca-visualization/)
if exe_mode == EXEMODE.ALL:
    pca = PCA()
    pcs = pca.fit_transform(data_pp.drop(['target'], axis=1))
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    fig = px.scatter_matrix(pcs, labels=labels, dimensions=range(4), color=data_pp["target"])
    fig.update_traces(diagonal_visible=False)
    fig.write_html("./results/pca_visualize.html")
    print(pca.explained_variance_ratio_)

####################
# 3. 데이터셋 활용 방안(train data 적음, balance 맞지 않음)
# 4. 불균형이 심한 데이터를 처리 및 학습하기 위한 방안
####################

input = data_pp.drop(['target'], axis=1).to_numpy()
target = data_pp['target'].to_numpy()

# 1) RandomOverSampler
# ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
# input_over, target_over = ros.fit_resample(input, target)
# print('Original dataset shape %s' % Counter(target))
# print('Resampled  dataset shape %s' % Counter(target_over))

# 2) Synthetic Minority Over-sampling Technique (SMOTE)
# TODO sampling_strategy = 'minority', 'not minority', 'not majority', 'all', 'auto'(='not majority')
# TODO n_neighbors = default 5
smote = SMOTE(sampling_strategy='auto', random_state=42)
input_over, target_over = smote.fit_resample(input, target)
print('Original dataset shape %s' % Counter(target))
print('Resampled  dataset shape %s' % Counter(target_over))

# 3) SMOTE-Nominal Continuous (SMOTENC)
# smotenc = SMOTENC(sampling_strategy='auto', random_state=42, categorical_features=[1, 2, 3])
# input_over, target_over = smotenc.fit_resample(input, target)
# print('Original dataset shape %s' % Counter(target))
# print('Resampled  dataset shape %s' % Counter(target_over))

# 4) ADAptive SYNthetic oversampling (ADASYN)
# adasyn = ADASYN(sampling_strategy='auto', random_state=42)
# input_over, target_over = adasyn.fit_resample(input, target)
# print('Original dataset shape %s' % Counter(target))
# print('Resampled  dataset shape %s' % Counter(target_over))

# 5) Borderline-SMOTE (B-SMOTE)


####################
# 5. 2종 이상의 모델 설계 및 성능 비교 : SVM, XGBoost, CatBoost, TODO MLP
# 6. 성능을 향상시키기 위한 각종 아이디어 - Ensemble
####################
train_input, valid_input, train_target, valid_target \
    = train_test_split(input_over, target_over, test_size=0.33, random_state=42)

# TODO 파라미터 최적화 : Scikit-learn GridSearchCV

# SVM
svc_model = SVC(kernel='rbf')
svc_model.fit(train_input, train_target)
pred_target = svc_model.predict(valid_input)
b_accr = balanced_accuracy_score(valid_target, pred_target)
print('##### SVM balanced accuracy: {}'.format(b_accr))

# XGBoost
xgb_model = XGBClassifier(booster='gbtree', max_depth=10, gamma=0.5, learning_rate=0.01, n_estimators=100,
                          random_state=99)
xgb_model = xgb_model.fit(train_input, train_target)
pred_target = xgb_model.predict(valid_input)
b_accr = balanced_accuracy_score(valid_target, pred_target)
print('##### XGB balanced accuracy: {}'.format(b_accr))

# CatBoost
cat_model = CatBoostClassifier()
cat_model.fit(train_input, train_target, use_best_model=True, early_stopping_rounds=100, verbose=100)
pred_target = cat_model.predict(valid_input)
b_accr = balanced_accuracy_score(valid_target, pred_target)
print('##### CB balanced accuracy: {}'.format(b_accr))

####################
# Save model
####################
