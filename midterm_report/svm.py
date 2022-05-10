from sklearn.svm import SVC
from _datasets import data_loader
import pandas as pd
import time

####################
# Research question
# 1. How does the choice of the kernel function and its parameter(s) influence the performance?
# 2. How does the choice of the penalty parameter influence the performance?
# 3. How many support vectors are obtained?
####################

# 학습/테스트 데이터 생성
train_data, train_label, test_data, test_label = data_loader.load()

####################
# RQ 4. kernel function and its parameter(s)
# - linear
# - poly
#   - degree(def=3), gamma(def=scale, auto), coef0(def=0.0)
# - rbf(def)
#   - gamma(def=scale, auto)
# - sigmoid
#   - gamma(def=scale, auto), coef0(def=0.0)
####################
# kernel_functions = [('linear',), ('poly', 3), ('poly', 5), ('poly', 7), ('poly', 9), ('rbf',), ('sigmoid',)]
# kernel_accr_list = []
# for kernel in kernel_functions:
#     kernel_name = None
#     if kernel[0] == 'poly':
#         kernel_name, degree = kernel
#         svc_model = SVC(kernel=kernel_name, degree=degree, verbose=True)
#         kernel_name += str(degree)
#     else:
#         svc_model = SVC(kernel=kernel[0], verbose=True)
#         kernel_name = kernel[0]
#
#     start_time = time.time()
#     svc_model.fit(train_data, train_label)
#     train_accr = svc_model.score(train_data, train_label)
#     test_accr = svc_model.score(test_data, test_label)
#     elapsed_time = time.time() - start_time
#
#     print('kernel: {}, train accuracy test accuracy: {}, elapsed time: {}'.format(kernel_name, train_accr, test_accr,
#                                                                                   elapsed_time))
#     kernel_accr_list.append((kernel_name, train_accr, test_accr, elapsed_time))
# df_kernel_accr = pd.DataFrame(kernel_accr_list,
#                               columns=['kernel function', 'train accuracy', 'test accuracy', 'elapsed time'])
# print(df_kernel_accr)
# df_kernel_accr.to_csv('./results/svm_kernel_accr.csv', index=False)


####################
# RQ 5&6. penalty parameter(C), number of support vectors
# - kernel function : rbf
####################
penalties = [1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3]
penalty_accr_list = []
for penalty in penalties:
    svc_model = SVC(kernel='rbf', C=penalty)

    start_time = time.time()
    svc_model.fit(train_data, train_label)
    num_sv = sum(svc_model.n_support_)
    accr = svc_model.score(test_data, test_label)
    elapsed_time = time.time() - start_time

    print('C: {}, accuracy: {}, support vectors: {}, elapsed time: {}'.format(penalty, accr,
                                                                              num_sv, elapsed_time))
    penalty_accr_list.append((penalty, accr, num_sv, elapsed_time))
df_penalty_accr = pd.DataFrame(penalty_accr_list,
                               columns=['C', 'accuracy', 'support vectors', 'elapsed time'])
print(df_penalty_accr)
df_penalty_accr.to_csv('./results/svm_rbf_penalty_accr.csv', index=False)
