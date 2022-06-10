import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Test data loding
feature_names = ['age', 'sex', 'education', 'marital_status', 'card_limit']
feature_names_use = ['use_' + str(i) for i in range(1, 7)]
feature_names_pay = ['pay_' + str(i) for i in range(1, 7)]
feature_names.extend(feature_names_use)
feature_names.extend(feature_names_pay)
feature_names_categorical = ['sex', 'education', 'marital_status']
feature_names_numerical = ['age', 'card_limit']
feature_names_numerical.extend(feature_names_use)
feature_names_numerical.extend(feature_names_pay)
data = pd.read_table('./_datasets/test_input.txt', delimiter=',', header=None, names=feature_names)

# Feature selection
np_use = data[feature_names_use].to_numpy()
np_pay = data[feature_names_pay].to_numpy()
# ① 연체회차 추가
np_overdue = np_use - np_pay
np_overdue[np_overdue > 0] = 1
np_overdue[np_overdue <= 0] = 0
data['overdue_num'] = np.sum(np_overdue, axis=1)
feature_names_numerical.extend(['overdue_num'])
# ④ 연체금액 추가
np_overdue_amt = np.sum(np_use - np_pay, axis=1)
np_overdue_amt[np_overdue_amt > 0] = 0
data['overdue_amt'] = abs(np_overdue_amt)
feature_names_numerical.extend(['overdue_amt'])

# Data scaling
scaler = StandardScaler()
data[feature_names_numerical] = scaler.fit_transform(data[feature_names_numerical])

# Model loding
model_fn = CatBoostClassifier()
model_fn.load_model('./models/catboost.model')

# Predict
test_input = data.to_numpy()
pred_target = model_fn.predict(test_input)

# Save result
pred_target.tofile('./results/test_output_limseonkyu.txt', sep='\n')
