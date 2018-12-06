import numpy as np

def ind(space, x):


# input x: continous
# e.g. std=1
def get_probs(space, x, std):
    probs = {}
    max_index = ind(space, x+3*std)
    min_index = ind(space, x-3*std)
    intervall=space[min_index:max_index]
    for s in intervall:
        probs.update({s : gaussian(s, x, std)})
    return probs

def gaussian(x, mean, std):
    return (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*np.square((x-mean)/std))


# Sample from given space with gaussian probabilities for given mean and sigma
space = np.linspace(-10, 10, 21)
mean = 5
sigma = 1
space_probabilities = np.array(gaussian(space, mean, sigma))
sample = np.random.choice(a=space, p=space_probabilities)
print(sample)

