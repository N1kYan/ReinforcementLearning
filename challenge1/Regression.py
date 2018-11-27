from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from Discretization import my_arctan


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
        regressorState = RandomForestRegressor(n_estimators=10, min_samples_split=2)

        old_state = env.reset()
        # Transform from 3 to 2 dim:(cos, sin, angular velocity) -> (angle, angular velocity)
        old_state = np.array([my_arctan(old_state[0], old_state[1]), old_state[2]])

        print("Regression: 0% ... ", end='')
        for i in range(epochs):

            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            # Transform from 3 to 2 dim:(cos, sin, angular velocity) -> (angle, angular velocity)
            next_state = np.array([my_arctan(next_state[0], next_state[1]), next_state[2]])

            rtx.append(np.append(old_state, action))
            rty.append(reward)
            stx.append(np.append(old_state, action))
            sty.append(next_state)

            if i % 10 == 0:  # 50 works nicely

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
                print("25% ... ", end='')
            if i == int(epochs * 0.5):
                print("50% ... ", end='')
            if i == int(epochs * 0.75):
                print("75% ... ", end='')

        print("Done!")

        # Plot loss curves
        plt.figure(0)
        plt.plot(plotr, label="Loss for reward fitting")
        plt.plot(plots, label="Loss for state fitting")
        plt.legend()
        plt.show()

        return regressorState, regressorReward