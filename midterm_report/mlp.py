from sklearn.neural_network import MLPClassifier

from _datasets import data_loader
import pandas as pd
import time

####################
# Research question
# 1. How does the choice of the structure influence the performance?
# 2. How does the choice of the activation function influence the performance?
# 3. How does the choice of the learning algorithm influence the performance?
####################

# train/test 데이터 생성
train_data, train_label, test_data, test_label = data_loader.load()
print(train_data.shape)
print(train_label.shape)

# 졍확도 평가를 위한 모델별 반복 train/test 횟수
num_iter = 10

####################
# RQ 1. structure
# - num of layers, num of neurons
####################
hidden_layers = [1, 2, 3]
hidden_neurons = [2, 4, 8, 16, 32, 64, 128, 256]
arch_accr_list = []
for layer in hidden_layers:
    for neuron in hidden_neurons:
        hidden_layer_sizes = (neuron, ) * layer
        mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                  activation='relu',
                                  solver="adam",
                                  early_stopping=True,
                                  max_iter=100)

        mean_accr = 0
        mean_elapsed_time = 0
        for index in range(num_iter):
            start_time = time.time()
            mlp_model = mlp_model.fit(train_data, train_label)
            accr = mlp_model.score(test_data, test_label)
            elapsed_time = time.time() - start_time
            print('hidden layrs: {}, hidden neurons: {}, accuracy: {}, elapsed time: {}'.format(layer, neuron, accr,
                                                                                                elapsed_time))
            mean_accr += accr / num_iter
            mean_elapsed_time += elapsed_time / num_iter
        arch_accr_list.append((layer, neuron, mean_accr, mean_elapsed_time))

# hidden layer/neuron 수 별 mean accuracy/elapsed time 저장
df_arch_accr = pd.DataFrame(arch_accr_list,
                            columns=['hidden layers', 'hidden neurons', 'mean accuracy', 'mean elapsed time'])
print(df_arch_accr)
df_arch_accr.to_csv('./results/mlp_arch_accr.csv', index=False)

####################
# RQ 1-1. epoch (layers: 2, neurons: 64)
####################
num_layers, num_neurons, num_epoch = 2, 64, 200
epoch_accr_list = []
for epoch in range(num_epoch):
    mlp_model = MLPClassifier(hidden_layer_sizes=(num_layers, num_neurons),
                              activation='relu',
                              solver='adam',
                              early_stopping=True,
                              max_iter=epoch + 1)
    mean_accr = 0
    for index in range(num_iter):
        mlp_model = mlp_model.fit(train_data, train_label)
        accr = mlp_model.score(test_data, test_label)
        print('epoch: {}, accuracy: {}'.format(epoch + 1, accr))

        mean_accr += accr / num_iter
    epoch_accr_list.append((epoch + 1, accr))

# epoch별 accuracy 저장
df_epoch_accr = pd.DataFrame(epoch_accr_list, columns=['epoch', 'mean accuracy'])
print(df_epoch_accr)
df_epoch_accr.to_csv('./results/mlp_epoch_accr_{}_{}_{}.csv'.format(num_epoch, num_layers, num_neurons), index=False)


####################
# RQ 2. activation function (layers: 2, neurons: 64)
# - logistic(sigmoid)
# - tanh
# - relu(def)
####################
activation_function = ['logistic', 'tanh', 'relu']
num_layers, num_neurons, num_epoch = 2, 64, 100
act_func_accr_list = []
for act_func in activation_function:
    for epoch in range(num_epoch):
        mlp_model = MLPClassifier(hidden_layer_sizes=(num_layers, num_neurons),
                                  activation=act_func,
                                  solver='adam',
                                  early_stopping=True,
                                  max_iter=epoch + 1)
        mean_accr = 0
        for index in range(num_iter):
            mlp_model = mlp_model.fit(train_data, train_label)
            accr = mlp_model.score(test_data, test_label)
            print('activation function: {}, epoch: {}, accuracy: {}'.format(act_func, epoch + 1, accr))
            mean_accr += accr / num_iter
        act_func_accr_list.append((act_func, epoch + 1, mean_accr))

df_act_func_accr = pd.DataFrame(act_func_accr_list, columns=['activation function', 'epoch', 'mean accuracy'])
print(df_act_func_accr)
df_act_func_accr.to_csv('./results/mlp_act_func_accr_{}_{}.csv'.format(num_layers, num_neurons), index=False)

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
learning_algorithm = ['lbfgs', 'nesterovs', 'adam']
num_layers, num_neurons, num_epoch = 2, 64, 100
lrn_alg_accr_list = []
for lrn_alg in learning_algorithm:
    for epoch in range(num_epoch):
        mlp_model = None
        if lrn_alg == 'lbfgs':
            mlp_model = MLPClassifier(hidden_layer_sizes=(num_layers, num_neurons),
                                      activation='relu',
                                      solver=lrn_alg,
                                      max_iter=epoch + 1)
        elif lrn_alg == 'nesterovs':
            mlp_model = MLPClassifier(hidden_layer_sizes=(num_layers, num_neurons),
                                      activation='relu',
                                      solver='sgd',
                                      learning_rate='constant',
                                      learning_rate_init=0.1,
                                      momentum=0.9,
                                      nesterovs_momentum=True,
                                      early_stopping=True,
                                      max_iter=epoch + 1)
        elif lrn_alg == 'adam':
            mlp_model = MLPClassifier(hidden_layer_sizes=(num_layers, num_neurons),
                                      activation='relu',
                                      solver=lrn_alg,
                                      learning_rate_init=0.1,
                                      early_stopping=True,
                                      max_iter=epoch + 1)

        mean_accr = 0
        for index in range(num_iter):
            mlp_model = mlp_model.fit(train_data, train_label)
            accr = mlp_model.score(test_data, test_label)
            print('learning algorithm: {}, epoch: {}, accuracy: {}'.format(lrn_alg, epoch + 1, accr))
            mean_accr += accr / num_iter
        lrn_alg_accr_list.append((lrn_alg, epoch + 1, mean_accr))

df_lrn_alg_accr = pd.DataFrame(lrn_alg_accr_list, columns=['learning algorithm', 'epoch', 'mean accuracy'])
print(df_lrn_alg_accr)
df_lrn_alg_accr.to_csv('./results/mlp_lrn_alg_accr_{}_{}.csv'.format(num_layers, num_neurons), index=False)
