import numpy as np
from tqdm import tqdm

class GTSS:
    def __init__(self, N, T):
        # lattice siza
        self.N = N
        # temperature list
        self.T = T
        # generate N*N spin state by random
        self.state = 2 * np.random.randint(0, 2, (self.N, self.N)) - 1

    # calc hamiltonian displacement　which is square lattice, after flipping one site(x,y)
    def calc_Hamiltonian_triangular(self, x, y):
        delta_H = 2 * self.state[x, y] * (self.state[(x+1)%self.N, y] + self.state[x, (y+1)%self.N] + self.state[(x-1)%self.N, y] + self.state[x, (y-1)%self.N]
                  +self.state[(x - 1)%self.N, (y + 1)%self.N] + self.state[(x + 1)%self.N, (y - 1)%self.N])
        return delta_H

    # flip random spin site
    def flip_spin(self):
        new_state = self.state.copy()
        [x, y] = np.random.randint(0, self.N, (2))
        new_state[x, y] *= -1
        return new_state, x, y

    # calc specious spin state by metropolis method
    def calc_spin_state(self, t):
        n = 10000
        for i in range(n):
            # get flip site
            new_state, x, y = self.flip_spin()
            # calc hamiltonian displacement
            deltaH = self.calc_Hamiltonian_triangular(x, y)
            # decide wheter to adopt
            if np.random.rand() <= np.exp(-deltaH/t):
                self.state = new_state

    def calc_each_temperature(self):
        # save spin state
        X = []
        T_crit = 3.64
        for t in tqdm(self.T):
            # init spin state
            self.state = 2 * np.random.randint(0, 2, (self.N, self.N)) - 1
            # generate spin state which temperature t
            self.calc_spin_state(t)
            # append generate spin state which temperature t
            X.append(self.state)
        Y = []
        for elm in np.where(self.T<=T_crit, 1, 0):
            Y.append([elm])
        return np.array(X), np.array(Y)