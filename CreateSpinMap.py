import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from module import SquareSpinState
from module import TriangularSpinState

class CSM:
    def __init__(self, N, T1_list, T2_list, npy_path, result_path):
        # lattice size
        self.N = N
        # temperature list
        self.T1_list = T1_list
        self.T2_list = T2_list
        # save path
        # train and test data
        self.npy_path = npy_path
        # spin map
        self.spin_map_path = result_path

    # make gif which write spin state transition
    def create_spinmap_gif(self, data, T, save_path):
        fig = plt.figure()
        data_size = len(T)
        frame = 50
        diff = data_size//frame

        def Plot(num):
            plt.cla()
            plt.imshow(data[num*diff])
            t_len = len(str(T[num*diff]))
            title = "temperature:" + str(T[num*diff]) + "0"*(5-t_len)
            plt.title(title)

        anim = FuncAnimation(fig, Plot, frames=frame)
        anim.save(save_path, writer='imagemagick')

    def createdata(self):
        # call module and generate spin map
        GSSS = SquareSpinState.GSSS(self.N, self.T1_list)
        GTSS = TriangularSpinState.GTSS(self.N, self.T2_list)

        # create spin map
        X_train, Y_train = GSSS.calc_each_temperature()
        X_test, Y_test = GTSS.calc_each_temperature()

        # create gif
        self.create_spinmap_gif(X_train, self.T1_list, self.spin_map_path + "squarelattice.gif")
        self.create_spinmap_gif(X_test, self.T2_list, self.spin_map_path + "triangularlattice.gif")

        # list2numpy and save file
        np.save(self.npy_path+"x_train.npy", X_train)
        np.save(self.npy_path+"y_train.npy", Y_train)
        np.save(self.npy_path+"x_test.npy", X_test)
        np.save(self.npy_path+"y_test.npy", Y_test)