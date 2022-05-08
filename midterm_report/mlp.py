from sklearn.neural_network import MLPClassifier

from _datasets import data_loader
import pandas as pd

####################
# Research question
# 1. How does the choice of the structure influence the performance?
# 2. How does the choice of the activation function influence the performance?
# 3. How does the choice of the learning algorithm influence the performance?
# 4. What is the optimal network structure? How is the 'optimality' defined?
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
hidden_layers = [3, 4, 5]
hidden_neurons = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
arch_accr_list = []
for layer in hidden_layers:
    for neuron in hidden_neurons:
        mlp_model = MLPClassifier(hidden_layer_sizes=(layer, neuron),
                                  activation='relu',
                                  solver="adam",
                                  early_stopping=True,
                                  max_iter=200)
        mean_accr = 0
        for index in range(num_iter + 1):
            mlp_model.fit(train_data, train_label)
            accr = mlp_model.score(test_data, test_label)
            print('hidden layrs: {}, hidden neurons: {}, accuracy: {}'.format(layer, neuron, accr))
            mean_accr += accr / num_iter
        arch_accr_list.append((layer, neuron, mean_accr))

df_arch_accr = pd.DataFrame(arch_accr_list, columns=['hidden layers', 'hidden neurons', 'mean accuracy'])
print(df_arch_accr)
df_arch_accr.to_csv('./results/mlp_arch_accr.csv', index=False)

####################
# RQ 2. activation function (layers: 5, neurons: 256)
# - logistic
# - tanh
# - relu(def)
####################
activation_function = ['logistic', 'tanh', 'relu']
num_layers, num_neurons = 5, 256
act_func_accr_list = []
for act_func in activation_function:
    mlp_model = MLPClassifier(hidden_layer_sizes=(num_layers, num_neurons),
                              activation=act_func,
                              solver="adam",
                              early_stopping=True,
                              max_iter=200)
    mean_accr = 0
    for index in range(num_iter + 1):
        mlp_model.fit(train_data, train_label)
        accr = mlp_model.score(test_data, test_label)
        print('activation function: {}, accuracy: {}'.format(act_func, accr))
        mean_accr += accr / num_iter
    act_func_accr_list.append((act_func, mean_accr))

df_acr_func_accr = pd.DataFrame(act_func_accr_list, columns=['activation function', 'mean accuracy'])
print(df_acr_func_accr)
df_acr_func_accr.to_csv('./results/mlp_act_func_accr.csv', index=False)

####################
# RQ 3. learning algorithm (layers: 5, neurons: 256)
# - lbfgs : the family of quasi-Newton methods
# - sgd : stochastic gradient descent
#   - learning_rate(def=constant, invscaling, adaptive), learning_rate_init(def=0.001)
#   - momentum(def=0.9), nesterovs_momentum(def=True), early_stopping(def=False)
# - adam
#   - learning_rate_init(def=0.001), early_stopping(def=False)
#   - beta_1(def=0.9), beta_2(def=0.999), epsilon(def=1e-8)
####################


####################
# RQ 4. optimal network structure
####################