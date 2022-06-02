import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier()
print(cat_model.get_all_params())

# file_names=['balanced_accuracy_oversampling_ADASYN.csv',
# 'balanced_accuracy_oversampling_RandomOverSampler.csv',
# 'balanced_accuracy_oversampling_SMOTE.csv',
# 'balanced_accuracy_oversampling_SMOTENC.csv']
#
# method = ['ADASYN', 'RandomOverSampler', 'SMOTE', 'SMOTENC']
#
# for index, file_name in enumerate(file_names):
#     result = pd.read_csv('./results/'+file_name, index_col=0)
#     result.plot(kind='line', x='number of samples', title='{} balanced accuracy'.format(method[index]))
#     plt.savefig('./results/balanced_accuracy_oversampling_{}.png'.format(method[index]))
