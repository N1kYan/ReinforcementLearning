from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

class Regressor:

    def __init__(self):
        None

    # Performs regression from given state and performed action
    # to successive state and observed reward
    def perform_regression(self, epochs, env):
        rtx = []
        rty = []
        stx = []
        sty = []
        plotr = []
        plots = []

        regressorReward = RandomForestRegressor(n_estimators=10, min_samples_split=2)
        regressorState = RandomForestRegressor(n_estimators=20, min_samples_split=2)

        old_state = env.reset()

        print("Regression: 0% ... ")
        for i in range(epochs):

            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            rtx.append(np.append(old_state, action))
            rty.append(reward)
            stx.append(np.append(old_state, action))
            sty.append(next_state)

            if i % 50 == 0:  # 50 works nicely

                regressorReward.fit(rtx, rty)
                fitrtx = regressorReward.predict(rtx)
                mse = mean_squared_error(rty, fitrtx)
                plotr.append(mse)

                regressorState.fit(stx, sty)
                fitstx = regressorState.predict(stx)
                mse = mean_squared_error(sty, fitstx)

                plots.append(mse)

            old_state = np.copy(next_state)

            if i == int(epochs * 0.25):
                print("25% ... ")
            if i == int(epochs * 0.5):
                print("50% ... ")
            if i == int(epochs * 0.75):
                print("75% ... ")

        print("Done!")

        # Plot loss curves
        plt.figure(0)
        plt.plot(plotr, label="Loss for reward fitting")
        plt.plot(plots, label="Loss for state fitting")
        plt.legend()
        plt.show()

        return regressorState, regressorReward