<h2> Deep Deterministic Policy Gradient </h2>
containing files:
<ul><li>DDPG.py: The main file for DDPG with the complete algorithm </li>
<li>NeuralNetworks.py: The file which creates the actor and the critic network</li>
<li>ReplayBuffer.py: Defines the memory of the algorithm with its own datastructure.</li>
<li>ActionNoise.py: Noise which is added to the chosen Action</li></ul>

<h4> Brief introduction to DDPG</h4>
DDPG is an reinforcement learning algorithm which combines the approaches of DPG and DQN. It is a model-free and off-policy algorithm and uses actor-critic methods with a deterministic target policy and deep Q-Learning. The critic is updated with the Bellman Equation with TD-error and the actor is updated using the DPG theorem. The replay buffer is used to sample random mini-batches to ensure independently and identically distributed date which are decorrelated. The target network is constrained to slow changes which greatly imporve stability.

<h4>Starting DDPG</h4>
1. Activating the virtual environment via <br/>
    <i>conda activate group19</i><br/>
2. Starting the main method with <br/>
<i> python3.6 DDPG.py</i>

<h4>Hyperparameter Performance </h4>
