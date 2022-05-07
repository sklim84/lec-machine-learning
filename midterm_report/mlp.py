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

# 학습/테스트 데이터 생성
train_data, train_label, test_data, test_label = data_loader.load()
print(train_data.shape)
print(train_label.shape)

# RQ 1. structure
# num of layers, num of neurons
hidden_layers = [3, 4, 5]
hidden_neurons = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

accuracy_list = []
for layer in hidden_layers:
    for neuron in hidden_neurons:
        mlp_model = MLPClassifier(hidden_layer_sizes=(layer, neuron),
                                  activation='relu',
                                  solver="adam",
                                  early_stopping=True,
                                  max_iter=200)
        mlp_model.fit(train_data, train_label)
        # pred_label = mlp_model.predict(test_data)
        mean_accucacy = mlp_model.score(test_data, test_label)
        print('hidden layrs: {}, hidden neurons: {}, mean accuracy: {}'.format(layer, neuron, mean_accucacy))
        accuracy_list.append((layer, neuron, mean_accucacy))

df_accuracy = pd.DataFrame(accuracy_list, columns=['hidden layers', 'hidden neurons', 'mean accuracy'])
print(df_accuracy)
df_accuracy.to_csv('./results/mlp_accuracy.csv', index=False)

# RQ 2. activation function (layers: 5, neurons: 256)
# - logistic
# - tanh
# - relu(def)

# RQ 3. learning algorithm (layers: , neurons: )
# - lbfgs : the family of quasi-Newton methods
# - sgd : stochastic gradient descent
#   - learning_rate(def=constant, invscaling, adaptive), learning_rate_init(def=0.001)
#   - momentum(def=0.9), nesterovs_momentum(def=True), early_stopping(def=False)
# - adam
#   - learning_rate_init(def=0.001), early_stopping(def=False)
#   - beta_1(def=0.9), beta_2(def=0.999), epsilon(def=1e-8)

# RQ 4. optimal network structure
