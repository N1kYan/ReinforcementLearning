import os

def find_directory(env_name, directory):

    for root, dirs, files in os.walk("./data"):
        print(root, dirs, files)

find_directory("BallBalancerSim-v0", "")