import os, sys
import numpy as np
import matplotlib.pyplot as plt
import keras

import CreateSpinMap
import LearnPhaseTransTemp

# save path
# train test data
npy_path = "./result/SpinState/"
# spin map
spin_map_path = "./result/SpinState/"
# model
model_path = "./model/model.h5"
# accurate
acc_path = "./result/acc/"

# lattice size
N = 32

# number of train data (square lattice)
count1 = 10000
# number of test data (triangle lattice)
count2 = 1000

def acc_plot(T, pred, LatticeType, T_crit):
    # initialize matplotlib.pyplot
    plt.cla()

    # plot option
    plt.plot(T, pred)
    plt.vlines([T_crit],-1, 101, "red", linestyles='dashed') # plot transrate point
    plt.title("Two-dimensional " + LatticeType + " lattice accurate")
    plt.xlabel("Temperature")
    plt.ylabel("accurate")
    plt.legend()
    plt.savefig(acc_path+LatticeType+"accurate")
    plt.show()

# call module that create spin map and learn data
#CSM = CreateSpinMap.CSM(count1, count2, N, npy_path, spin_map_path)
#CSM.createdata()

#LPTT = LearnPhaseTransTemp.LPTT(N, npy_path, model_path, learning_rate=1e-3, l2_const=1e-4, verbose=1, epochs=100, batch_size=36)
#LPTT.learndata()

# load train and test data
X_train = np.load(npy_path+"x_train.npy")
X_train = X_train.reshape(X_train.shape+(1,))
Y_train = np.load(npy_path+"y_train.npy")
X_test = np.load(npy_path+"x_test.npy")
X_test = X_test.reshape(X_test.shape+(1,))
Y_test = np.load(npy_path+"y_test.npy")

# load model data
model = keras.models.load_model(model_path)
print(model.evaluate(X_test, Y_test)[1], "%")

T1_list = np.linspace(0.1, 5.5, count1).round(3)
T2_list = np.linspace(2.0, 7.5, count2).round(3)
# text result
pred = model.predict(X_test)
pred = pred.reshape(count2)
print((pred*100).astype(int))
acc_plot(T2_list, pred*100, "triangular", 3.64)

# train result
pred = model.predict(X_train)
pred = pred.reshape(count1)
acc_plot(T1_list, pred*100, "square", 2.27)