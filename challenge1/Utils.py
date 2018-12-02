import pickle
import matplotlib.pyplot as plt


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def visualize(value_function, policy):
    plt.figure()
    plt.title("Value function")
    plt.imshow(value_function)
    plt.colorbar()
    plt.show()
