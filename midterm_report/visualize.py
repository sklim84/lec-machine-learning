import pandas as pd
import matplotlib.pyplot as plt

# ###################
# RQ 1-1. epoch (layers: 2, neurons: 64)
# ###################
df_mlp_epoch_accr = pd.read_csv('./results/mlp_epoch_accr_300_2_64.csv')
print(df_mlp_epoch_accr)

mean_accr_list = df_mlp_epoch_accr['mean accuracy'].to_list()
epochs = list(range(1, 301))

plt.title('2 hidden layers and 64 neurons')
plt.xlabel('epochs')
plt.xticks([1, 100, 200, 300])
plt.ylabel('Test Accuracy')
plt.ylim(0, 1.0)
plt.plot(epochs, mean_accr_list)
plt.show()


####################
# RQ 2. activation function (layers: 2, neurons: 64)
# - logistic(sigmoid)
# - tanh
# - relu(def)
####################
df_act_func_accr = pd.read_csv('./results/mlp_act_func_accr_2_64.csv')
print(df_act_func_accr)

sigmoid_accr_list = df_act_func_accr.iloc[:100, :]['mean accuracy'].to_list()
tanh_accr_list = df_act_func_accr.iloc[100:200, :]['mean accuracy'].to_list()
relu_accr_list = df_act_func_accr.iloc[200:, :]['mean accuracy'].to_list()
epochs = list(range(1, 101))

plt.xlabel('epochs')
plt.xticks([1, 25, 50, 75, 100])
plt.ylabel('Test Accuracy')
plt.ylim(0, 1.0)
plt.plot(epochs, sigmoid_accr_list, label='sigmoid')
plt.plot(epochs, tanh_accr_list, label='tanh')
plt.plot(epochs, relu_accr_list, label='relu')
plt.legend(loc='lower right')
plt.show()


####################
# RQ 3. learning algorithm (layers: 2, neurons: 64)
# - lbfgs : the family of quasi-Newton methods
# - sgd : stochastic gradient descent
#   - learning_rate(def=constant, invscaling, adaptive), learning_rate_init(def=0.001)
#   - momentum(def=0.9), nesterovs_momentum(def=True), early_stopping(def=False)
# - adam
#   - learning_rate_init(def=0.001), early_stopping(def=False)
#   - beta_1(def=0.9), beta_2(def=0.999), epsilon(def=1e-8)
####################
df_lrn_alg_accr = pd.read_csv('./results/mlp_lrn_alg_accr_2_64.csv')
print(df_lrn_alg_accr)

lbfgs_accr_list = df_lrn_alg_accr.iloc[:100, :]['mean accuracy'].to_list()
nesterovs_accr_list = df_lrn_alg_accr.iloc[100:200, :]['mean accuracy'].to_list()
adam_accr_list = df_lrn_alg_accr.iloc[200:, :]['mean accuracy'].to_list()
epochs = list(range(1, 101))

plt.xlabel('epochs')
plt.xticks([1, 25, 50, 75, 100])
plt.ylabel('Test Accuracy')
plt.ylim(0, 1.0)
plt.plot(epochs, lbfgs_accr_list, label='lbfgs')
plt.plot(epochs, nesterovs_accr_list, label='sgd')
plt.plot(epochs, adam_accr_list, label='adam')
plt.legend(loc='lower right')
plt.show()


####################
# RQ 3-1. learning rate(layers: 2, neurons: 64)
# - learning algorithm : adam
####################
df_adam_accr = pd.read_csv('./results/mlp_adam_accr_2_64_iter1.csv')
print(df_adam_accr)

lr_1e_1_list = df_adam_accr.iloc[:100, :]['mean accuracy'].to_list()
lr_1e_2_list = df_adam_accr.iloc[100:200, :]['mean accuracy'].to_list()
lr_1e_3_list = df_adam_accr.iloc[200:, :]['mean accuracy'].to_list()
epochs = list(range(1, 101))

plt.xlabel('epochs')
plt.xticks([1, 25, 50, 75, 100])
plt.ylabel('Test Accuracy')
plt.ylim(0, 1.0)
plt.plot(epochs, lr_1e_1_list, label='lr=0.1')
plt.plot(epochs, lr_1e_2_list, label='lr=0.01')
plt.plot(epochs, lr_1e_3_list, label='lr=0.001')
plt.legend(loc='lower right')
plt.show()


####################
# RQ 5&6. penalty parameter(C), number of support vectors
# - kernel function : rbf
####################
df_rbf_penalty_accr = pd.read_csv('./results/svm_rbf_penalty_accr.csv')
print(df_rbf_penalty_accr)

rbf_penalty_accr = df_rbf_penalty_accr['accuracy'].to_list()
rbf_penalty_accr_num_sv = df_rbf_penalty_accr['support vectors'].to_list()
penalties = [1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3]

plt.xlabel('Penalty parameter(C)')
plt.xscale('log')
plt.ylabel('Test Accuracy')
plt.ylim(0, 1.0)
plt.plot(penalties, rbf_penalty_accr)
plt.show()

plt.xlabel('Penalty parameter(C)')
plt.xscale('log')
plt.ylabel('Number of support vectors')
plt.plot(penalties, rbf_penalty_accr_num_sv, color='g')
plt.show()
