from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np

X = np.loadtxt("hyperparameters.csv", delimiter=",")
y = np.loadtxt("clust_metrics.csv", delimiter=",")
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=101)
'''hidden_layer_counts = [n for n in range(1, 6)]
hidden_layer_sizes = [5+(5*n) for n in range(8)]
learning_rates=[0.001*(2**(-2+n)) for n in range(5)]'''

hidden_layer_counts = [4]
hidden_layer_sizes = [40]
learning_rates = [0.004]


for num_lay in hidden_layer_counts:
    for lay_size in hidden_layer_sizes:
        for learning_rate in learning_rates:
            layers = [lay_size for i in range(num_lay)]
            model = MLPRegressor(layers, random_state=101,verbose=False, learning_rate_init=learning_rate, activation="tanh", batch_size=32, early_stopping = True).fit(X_train, y_train)
            print("("+str(layers)+", "+str(learning_rate)+")")
            print(model.score(X_test, y_test)) #R^2

frac_accepts = [0.01*(n+1) for n in range(9, 50)]
dist_weights = [0.025*(n+1) for n in range(11, 40)]
relaxation_factors = [1+(0.01*n) for n in range(51)]

hyp = []
perf = []

for x in frac_accepts:
    for y in dist_weights:
        for z in relaxation_factors:
            hyp.append([x,y,z])
            perf.append(model.predict([[x, y, z]])[0])

np.savetxt("nn_input.csv", np.array(hyp), delimiter=",")
np.savetxt("nn_output.csv", np.array(perf), delimiter=",")
