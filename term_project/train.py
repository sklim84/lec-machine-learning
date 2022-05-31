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
import os


# 실행모드
class EXEMODE(Enum):
    ALL = 0
    FINAL = 1


# 데이터 분포 확인
def data_distribution(data, type='numerical', about='base'):
    num_col = 3
    num_row = int(np.ceil(len(data.columns) / num_col))
    fig, axs = plt.subplots(num_row, num_col)
    fig.set_figheight(num_row * 2.5)
    fig.set_figwidth(num_col * 4)
    for index, feature in enumerate(data.columns):
        i_row = int(np.floor((index / num_col) % num_row))
        i_col = index % num_col
        ax = axs[i_col] if num_row == 1 else axs[i_row, i_col]

        if type == 'numerical':
            data[feature].plot.hist(ax=ax, title=feature, alpha=0.5)
        elif type == 'categorical':
            data[feature].value_counts().plot(ax=ax, kind='bar', title=feature, alpha=0.5)
    plt.tight_layout()
    plt.savefig('./results/data_distribution_{}_{}.png'.format(about, type))


# 데이터 전처리
def data_preprocessing(data):
    counter = Counter(data['target'])
    print('##### number of data before preprocessing: {} (0: {}, 1: {})'.format(sum(counter.values()), counter[0],
                                                                                counter[1]))
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

    counter = Counter(data_pp['target'])
    print('##### number of data after preprocessing: {} (0: {}, 1: {})'.format(sum(counter.values()), counter[0],
                                                                               counter[1]))
    return data_pp


# 기본 모델(SVM, XGBoost, CatBoost) 학습
def training_default_model(input, target, about='base', save_file=True):
    train_input, valid_input, train_target, valid_target \
        = train_test_split(input, target, test_size=0.33, random_state=42)
    # SVM
    svc_model = SVC()
    svc_model.fit(train_input, train_target)
    svc_pred_target = svc_model.predict(valid_input)
    svc_b_accr = balanced_accuracy_score(valid_target, svc_pred_target)
    # XGBoost
    xgb_model = XGBClassifier(random_state=42)
    xgb_model = xgb_model.fit(train_input, train_target)
    xgb_pred_target = xgb_model.predict(valid_input)
    xgb_b_accr = balanced_accuracy_score(valid_target, xgb_pred_target)
    # CatBoost
    cat_model = CatBoostClassifier()
    cat_model.fit(train_input, train_target, verbose=100)
    cat_pred_target = cat_model.predict(valid_input)
    cat_b_accr = balanced_accuracy_score(valid_target, cat_pred_target)

    result = pd.DataFrame(columns=['model', 'balanced accuracy'])
    result.loc[0] = [svc_model.__class__.__name__, svc_b_accr]
    result.loc[1] = [xgb_model.__class__.__name__, xgb_b_accr]
    result.loc[2] = [cat_model.__class__.__name__, cat_b_accr]
    print('##### Model training result ({})'.format(about))
    print(result)
    if save_file:
        result.to_csv('./results/balanced_accuracy_{}.csv'.format(about))
    return result


# feature 중요도 계산
def feature_importance(model, x, y, feature_names, postfix=None):
    if postfix == None:
        postfix = model.__class__.__name__

    # permutation importance
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.33, random_state=42)
    model.fit(train_x, train_y)
    print('complete model fit...')
    perm_importance = permutation_importance(model, valid_x, valid_y, n_repeats=30, random_state=0, n_jobs=3)
    sorted_idx = perm_importance.importances_mean.argsort()
    print('complete permutation importance...')

    # save csv
    result = pd.DataFrame()
    result['feature'] = [feature_names[index] for index in sorted_idx]
    result['importance'] = perm_importance.importances_mean[sorted_idx]
    result.sort_values(by='importance', ascending=False, inplace=True)
    result.to_csv('./results/feature_importance_{}.csv'.format(postfix))
    print(result)

    # save plot
    sns.set(rc={"axes.unicode_minus": False}, style='whitegrid')
    fig, ax = plt.subplots()
    ax.boxplot(perm_importance.importances[sorted_idx].T, vert=False,
               labels=[feature_names[index] for index in sorted_idx])
    ax.set_title("Feature Importance ({})".format(postfix))
    fig.tight_layout()
    plt.savefig('./results/feature_importance_{}.png'.format(postfix))


# 실행모드
exe_mode = EXEMODE.FINAL

####################
# 1. Data analysis
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
    data_distribution(data_base[feature_names_numerical], type='numerical', about='base')
    data_distribution(data_base[feature_names_categorical], type='categorical', about='base')
    # base 데이터 기반 기본 모델 학습
    input = data_base.drop(['target'], axis=1).to_numpy()
    target = data_base['target'].to_numpy()
    training_default_model(input, target, about='base')

####################
# 2. Data preprocessing
####################

# 1) Outlier data processing
if exe_mode == EXEMODE.FINAL:
    data_pp = data_preprocessing(data_base)

    input = data_pp.drop(['target'], axis=1).to_numpy()
    target = data_pp['target'].to_numpy()
    # training_default_model(input, target, about='preprocessed')

# 2) Feature scaling
if exe_mode == EXEMODE.FINAL:
    scaler = StandardScaler()
    data_pp[feature_names_numerical] = scaler.fit_transform(data_pp[feature_names_numerical])
    data_distribution(data_pp[feature_names_numerical], type='numerical', about='scaled')

    input = data_pp.drop(['target'], axis=1).to_numpy()
    target = data_pp['target'].to_numpy()
    # training_default_model(input, target, about='scaled')

# 3) Feature 중요도 평가
if exe_mode == EXEMODE.ALL:
    input = data_pp.drop(['target'], axis=1).to_numpy()
    target = data_pp['target'].to_numpy()
    # 기본 모델(SVM, XGBoost, CatBoost) 이용
    feature_importance(SVC(), input, target, feature_names)
    feature_importance(XGBClassifier(random_state=42), input, target, feature_names)
    feature_importance(CatBoostClassifier(), input, target, feature_names)

# 4) Feature selection : 삭제(성별, 최종학력, 결혼여부)
if exe_mode == EXEMODE.ALL:
    data_fs = data_pp.copy()

    data_fs.drop(['sex'], axis=1, inplace=True)
    data_fs.drop(['education'], axis=1, inplace=True)
    data_fs.drop(['marital_status'], axis=1, inplace=True)

    input = data_fs.drop(['target'], axis=1).to_numpy()
    target = data_fs['target'].to_numpy()
    training_default_model(input, target, about='feature_selection')

# 5) Feature extraction (PCA)
# 시각화 : https://plotly.com/python/pca-visualization/
# 참고 : Scale에 따라 주성분의 설명 가능한 분산량이 왜곡될 수 있기 때문에 PCA 수행 전 표준화 필요
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

####################
# 3. Data balancing
# - 데이터셋 활용 방안(train data 적음, balance 맞지 않음)
# - 불균형이 심한 데이터를 처리 및 학습하기 위한 방안
####################
if exe_mode == EXEMODE.ALL:
    result_columns = ['number of samples', 'SVM', 'XGBoost', 'CatBoost']
    result_over_total = {'RandomOverSampler': pd.DataFrame(columns=result_columns),
                         'SMOTE': pd.DataFrame(columns=result_columns),
                         'SMOTENC': pd.DataFrame(columns=result_columns),
                         'ADASYN': pd.DataFrame(columns=result_columns)}

    input = data_pp.drop(['target'], axis=1).to_numpy()
    target = data_pp['target'].to_numpy()
    num_samples = list(range(20000, 100001, 10000))
    num_samples.append(0, Counter(data_pp['target'])[0])
    for num_sample in num_samples:
        sampling_strategy = {0: num_sample, 1: num_sample}
        oversampling_methods = [RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42),
                                SMOTE(sampling_strategy=sampling_strategy, random_state=42),
                                SMOTENC(sampling_strategy=sampling_strategy, random_state=42,
                                        categorical_features=[1, 2, 3]),
                                ADASYN(sampling_strategy=sampling_strategy, random_state=42)]
        for oversampling_method in oversampling_methods:
            method = oversampling_method.__class__.__name__
            input_over, target_over = oversampling_method.fit_resample(input, target)

            print('##### Oversampling method: {}'.format(method))
            # ADASYN : 클래스별 데이터 생성 수 상이
            print('##### Resampled dataset shape %s' % Counter(target_over))
            b_accr = training_default_model(input_over, target_over,
                                            about='over_{}_{}'.format(method, len(target_over)), save_file=False)
            result_total = result_over_total.get(method)
            item = pd.Series([len(target_over), b_accr.iloc[0, 1], b_accr.iloc[1, 1], b_accr.iloc[2, 1]],
                             index=result_total.columns)
            result_total = result_total.append(item, ignore_index=True)
            result_over_total.update({method: result_total})

    for result in result_over_total.items():
        result[1]['number of samples'] = result[1]['number of samples'].astype('int')
        result[1].sort_values(by='number of samples', inplace=True)
        result[1].reset_index(drop=True, inplace=True)
        result[1].plot(kind='line', x='number of samples', title='{} balanced accuracy'.format(result[0]))
        plt.savefig('./results/balanced_accuracy_oversampling_{}.png'.format(result[0]))
        result[1].to_csv('./results/balanced_accuracy_oversampling_{}.csv'.format(result[0]))

####################
# 5. 2종 이상의 모델 설계 및 성능 비교 : SVM, XGBoost, CatBoost, TODO MLP
# 6. 성능을 향상시키기 위한 각종 아이디어 - Ensemble
####################
# train_input, valid_input, train_target, valid_target = train_test_split(input_over, target_over, test_size=0.33, random_state=42)

# TODO 파라미터 최적화 : Scikit-learn GridSearchCV

# SVM

# MLP

# XGBoost
# xgb_model = XGBClassifier(booster='gbtree', max_depth=10, gamma=0.5, learning_rate=0.01, n_estimators=100,
#                           random_state=99)

# CatBoost
# cat_model = CatBoostClassifier()
# cat_model.fit(train_input, train_target, use_best_model=True, early_stopping_rounds=100, verbose=100)

# TODO 성능개선 : feature selection
# feature selection : 추가(이용금액, 납부금액, 연체회차, 연체금액), 삭제(과거 6개월간 청구대금, 납부금액)
# if exe_mode == EXEMODE.ALL:
# np_use = data_fs[feature_names_use].to_numpy()
# np_pay = data_fs[feature_names_pay].to_numpy()
# # 연체회차 추가
# np_overdue = np_use - np_pay
# np_overdue[np_overdue > 0] = 1
# np_overdue[np_overdue <= 0] = 0
# data_fs['overdue_num'] = np.sum(np_overdue, axis=1)
# feature_names.extend(['overdue_num'])
# feature_names_numerical.extend(['overdue_num'])
# # 납부금액 추가
# np_pay = data_fs[feature_names_pay].to_numpy()
# data_fs['pay_amt'] = np.sum(np_pay, axis=1)
# feature_names.extend(['pay_amt'])
# feature_names_numerical.extend(['pay_amt'])
# # 이용금액 추가
# data_fs['use_amt'] = np.sum(np_use, axis=1)
# feature_names.extend(['use_amt'])
# feature_names_numerical.extend(['use_amt'])
# # 연체금액 추가
# np_overdue_amt = np.sum(np_use - np_pay, axis=1)
# np_overdue_amt[np_overdue_amt > 0] = 0
# data_fs['overdue_amt'] = abs(np_overdue_amt)
# feature_names.extend(['overdue_amt'])
# feature_names_numerical.extend(['overdue_amt'])
#
# # 과거 6개월간 청구대금/납부금액 삭제
# data_fs.drop(feature_names_use, axis=1, inplace=True)
# data_fs.drop(feature_names_pay, axis=1, inplace=True)
# feature_names_numerical = [ele for ele in feature_names_numerical if ele not in feature_names_use]
# feature_names_numerical = [ele for ele in feature_names_numerical if ele not in feature_names_pay]


####################
# Save model
####################
