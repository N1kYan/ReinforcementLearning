<h2> Deep Deterministic Policy Gradient </h2>

<ul><li>DDPG.py: The main file for DDPG with the complete algorithm </li>
<li>NeuralNetworks.py: The file which creates the actor and the critic network</li>
<li>ReplayBuffer.py: Defines the memory of the algorithm with its own datastructure</li>
<li>ActionNoise.py: Noise which is added to the chosen action</li></ul>

<h4> Brief introduction to DDPG</h4>
DDPG is an reinforcement learning algorithm which combines the approaches of DPG and DQN. It is a model-free and off-policy algorithm and uses actor-critic methods with a deterministic target policy and deep Q-Learning. The critic is updated with the Bellman Equation with TD-error and the actor is updated using the DPG theorem. The replay buffer is used to sample random mini-batches to ensure independently and identically distributed date which are decorrelated. The target network is constrained to slow changes which greatly imporve stability.

<h4>Starting DDPG</h4>
1. Activating the virtual environment via <br/>
    <i>conda activate group19</i><br/>
2. Starting the main method with <br/>
<i> python3.6 DDPG.py</i>

<h4>Hyperparameter Performance </h4>
1. To change the deep neural network structures, please edit NeuralNetworks.py. Setting num_layers=1 automaticall uses one hidden layer with fc1_units hidden units. <br/>
2. Replay Buffer specifications can be directly changed in line #326 in main() method of DDPG.py. E.g. one could change the batch_size here to 1024 or back to 64, aswell as the total buffer size. <br/>
3. The action noise object has to be initialized in main() method of DDPG.py. Hyperparameters can also be set there. E.g. might set the standard deviation sigma of the gaussion noise to 0.1 in line #323. To implement different noises, pleasy directly modifiy AcitonNoise.py <br/>
4. The remaining hyperparameters are defined in the call of the training() method in line #330 in the main() method. One could run the algorithm with less discounting by changing to gamma=x in line #332 <br/>
