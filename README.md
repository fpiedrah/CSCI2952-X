## GENERATING TRAJECTORIES DATASET

The generate_trajectories function trains a reinforcement learning (RL) algorithm and collects trajectory data from a specified environment at regular intervals.

### PARAMETERS

- **algorithm (BaseAlgorithm):** The RL algorithm to train and use for generating trajectories (e.g., PPO, DQN).
- **environment (VecEnv):** The vectorized environment in which the algorithm interacts.
- **training_steps (int):** The total number of training steps to perform.
- **rollout_interval (int):** Number of training steps between data collection intervals.
- **rollout_steps (int):** Number of steps to perform during each data collection rollout.
- **rollout_trajectories (int):** Number of trajectories to collect per rollout.

### RETURNS

- **list[dict[str, Any]]:** A list of dictionaries containing collected trajectory data.

### 

```json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create a vectorized environment
env = make_vec_env('CartPole-v1', n_envs=4)

# Initialize the RL algorithm
model = PPO('MlpPolicy', env)

# Generate trajectories
dataset = generate_trajectories(
    algorithm=model,
    environment=env,
    training_steps=10000,
    rollout_interval=1000,
    rollout_steps=200,
    rollout_trajectories=5,
)

# The dataset now contains trajectory data collected during training
print(f"Collected {len(dataset)} data points")
```
