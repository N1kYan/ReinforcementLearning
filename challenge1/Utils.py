import pickle
import matplotlib.pyplot as plt

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def print_array(arr):
    for i in arr:
        print(i)


def visualize(value_function, policy, disc=None):
    plt.figure()
    plt.title("Value function")
    plt.imshow(value_function)
    plt.colorbar()

    if disc is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(disc.state_space_size[0]), labels=disc.state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(disc.state_space_size[1]), labels=disc.state_space[1].round(1))

    plt.show()

    plt.title("Policy")
    plt.imshow(policy)
    plt.colorbar()

    if disc is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(disc.state_space_size[0]), labels=disc.state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(disc.state_space_size[1]), labels=disc.state_space[1].round(1))

    plt.show()