from sklearn.svm import SVC
from _datasets import data_loader
import pandas as pd

####################
# Research question
# 1. How does the choice of the kernel function and its parameter(s) influence the performance?
# 2. How does the choice of the penalty parameter influence the performance?
# 3. How many support vectors are obtained?
####################

# 학습/테스트 데이터 생성
train_data, train_label, test_data, test_label = data_loader.load()

# RQ 1. kernel function and its parameter(s)
# - linear
# - poly
#   - degree(def=3), gamma(def=scale, auto), coef0(def=0.0)
# - rbf(def)
#   - gamma(def=scale, auto)
# - sigmoid
#   - gamma(def=scale, auto), coef0(def=0.0)
# - precomputed
svc_model = SVC(kernel='rbf')
svc_model.fit(train_data, train_label)
mean_accuracy = svc_model.score(test_data, test_label)
print(mean_accuracy)
# pred_label = svc_model.predict(test_data)

# RQ 2. penalty parameter
# - class_weight : dict or ‘balanced’, default=None

# RQ 3. num of support vectors
# TODO 각 결과별 support vector 수, 의미해석
