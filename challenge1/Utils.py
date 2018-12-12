import pickle
import matplotlib.pyplot as plt

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def print_array(arr):
    for i in arr:
        print(i)


def visualize(value_function, policy, R, state_distribution, state_space=None):
    plt.figure()
    plt.title("Value function")
    plt.imshow(value_function)
    plt.colorbar()
    if state_space is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(len(state_space[0])), labels=state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(len(state_space[1])), labels=state_space[1].round(1))

    plt.figure()
    plt.title("Policy")
    plt.imshow(policy)
    plt.colorbar()
    if state_space is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(len(state_space[0])), labels=state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(len(state_space[1])), labels=state_space[1].round(1))

    plt.figure()
    plt.imshow(R)
    plt.title("Reward function")
    plt.colorbar()
    if state_space is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(len(state_space[0])), labels=state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(len(state_space[1])), labels=state_space[1].round(1))

    plt.figure()
    plt.imshow(state_distribution)
    plt.title("State distribution after evaluating")
    if state_space is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(len(state_space[0])), labels=state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(len(state_space[1])), labels=state_space[1].round(1))

    plt.show()
