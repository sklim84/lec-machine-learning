import pandas as pd
import os
import matplotlib.pyplot as plt

result_columns = ['number of samples', 'SVM', 'XGBoost', 'CatBoost']
result_over_total = {'RandomOverSampler': pd.DataFrame(columns=result_columns),
                     'SMOTE': pd.DataFrame(columns=result_columns),
                     'SMOTENC': pd.DataFrame(columns=result_columns)}
for file_name in os.listdir('./results'):
    prefix = 'balanced_accuracy_over_'
    if file_name.startswith(prefix):
        b_accr = pd.read_csv('./results/' + file_name)
        # os.remove('./results/' + file_name)
        file_name = os.path.splitext(file_name)[0]
        file_name = file_name.replace(prefix, '')
        elements = file_name.split('_')
        method = elements[0]
        num_sample = elements[1]
        result = result_over_total.get(method)
        item = pd.Series([num_sample, b_accr.iloc[0, 2], b_accr.iloc[1, 2], b_accr.iloc[0, 2]],
                         index=result.columns)
        result = result.append(item, ignore_index=True)
        result_over_total.update({method: result})

for result in result_over_total.items():
    result[1]['number of samples'] = result[1]['number of samples'].astype('int')
    result[1].sort_values(by='number of samples', inplace=True)
    result[1].reset_index(drop=True, inplace=True)
    print(result[1])
    result[1].plot(kind='line', x='number of samples', title='{} balanced accuracy'.format(result[0]))
    plt.show()
    plt.savefig('./results/balanced_accuracy_oversampling_{}.png'.format(result[0]))
    result[1].to_csv('./results/balanced_accuracy_oversampling_{}.csv'.format(result[0]))
