import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from module import SquareSpinState
from module import TriangularSpinState

class CSM:
    def __init__(self, count1, count2, N, npy_path, result_path):
        # count1 is num of train data
        self.count1 = count1
        # count2 is num of test data
        self.count2 = count2
        # lattice size
        self.N = N
        # save path
        # train and test data
        self.npy_path = npy_path
        # spin map
        self.spin_map_path = result_path

    # make gif which write spin state transition
    def create_spinmap_gif(self, data, T, save_path):
        fig = plt.figure()

        def Plot(num):
            plt.cla()
            plt.imshow(data[num])
            t_len = len(str(T[num]))
            title = "temperature:" + str(T[num]) + "0"*(5-t_len)
            plt.title(title)

        anim = FuncAnimation(fig, Plot, frames=100)
        anim.save(save_path, writer='imagemagick')

    def createdata(self):
        # temperature list
        T1_list = np.linspace(0.1, 5.5, self.count1).round(3)
        T2_list = np.linspace(2.0, 7.5,self.count2).round(3)

        # call module and generate spin map
        GSSS = SquareSpinState.GSSS(self.N, T1_list)
        GTSS = TriangularSpinState.GTSS(self.N, T2_list)

        # create spin map
        X_train, Y_train = GSSS.calc_each_temperature()
        X_test, Y_test = GTSS.calc_each_temperature()

        # create gif
        self.create_spinmap_gif(X_train, T1_list, self.spin_map_path + "squarelattice.gif")
        self.create_spinmap_gif(X_test, T2_list, self.spin_map_path + "triangularlattice.gif")

        # list2numpy and save file
        np.save(self.npy_path+"x_train.npy", X_train)
        np.save(self.npy_path+"y_train.npy", Y_train)
        np.save(self.npy_path+"x_test.npy", X_test)
        np.save(self.npy_path+"y_test.npy", Y_test)