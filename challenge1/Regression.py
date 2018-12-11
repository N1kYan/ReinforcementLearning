from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pickle
from Utils import *
import sys

class Regressor:

    def __init__(self):
        self.regressorReward = RandomForestRegressor(n_estimators=10, min_samples_split=2)
        self.regressorState = RandomForestRegressor(n_estimators=20, min_samples_split=2)

    # Performs regression from given state and performed action
    # to successive state and observed reward
    def perform_regression(self, epochs, env, save_flag):

        if not save_flag and open('./pickle/reg.pkl'):
            print()
            print("Found regression file.")
            print()
            with open('./pickle/reg.pkl', 'rb') as pickle_file:
                (rS, rR) = pickle.load(pickle_file)
            self.regressorState = rS
            self.regressorReward = rR
        else:
            rtx = []
            rty = []
            stx = []
            sty = []
            plotr = []
            plots = []
            old_state = env.reset()

            print("Regression: 0% ... ", end='')
            sys.stdout.flush()

            for i in range(epochs):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)

                #rtx.append(np.append(old_state, action))
                rtx.append(next_state)
                rty.append(reward)
                stx.append(np.append(old_state, action))
                sty.append(next_state)

                if i % 100 == 0:  # 50 works nicely

                    # Regression from next observed state to observed reward
                    self.regressorReward.fit(rtx, rty)
                    fitrtx = self.regressorReward.predict(rtx)
                    mse = mean_squared_error(rty, fitrtx)
                    plotr.append(mse)

                    # Regression from state and action to next state
                    self.regressorState.fit(stx, sty)
                    fitstx = self.regressorState.predict(stx)
                    mse = mean_squared_error(sty, fitstx)

                    plots.append(mse)

                old_state = np.copy(next_state)

                if i == int(epochs * 0.25):
                    print("25% ... ", end='')
                    sys.stdout.flush()
                if i == int(epochs * 0.5):
                    print("50% ... ", end='')
                    sys.stdout.flush()
                if i == int(epochs * 0.75):
                    print("75% ... ", end='')
                    sys.stdout.flush()

            print("Done!")
            print()
            sys.stdout.flush()

            # Plot loss curves
            plt.figure(0)
            plt.plot(plotr, label="Loss for reward fitting")
            plt.plot(plots, label="Loss for state fitting")
            plt.legend()
            plt.show()

            print("Saving regression file.")
            print()
            save_object((self.regressorState, self.regressorReward), './pickle/reg.pkl')