import typing
import uuid

import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def generate_trajectories(
    algorithm: BaseAlgorithm,
    environment: VecEnv,
    training_steps: int,
    rollout_interval: int,
    rollout_steps: int,
    rollout_trajectories: int,
) -> list[dict[str, typing.Any]]:
    """
    Train an RL algorithm and collect trajectory data at specified intervals.

    Parameters:
        algorithm (BaseAlgorithm): The reinforcement learning algorithm to train.
        environment (VecEnv): The environment for training and data collection.
        training_steps (int): Total number of training steps.
        rollout_interval (int): Steps between data collection intervals.
        rollout_steps (int): Steps per data collection rollout.
        rollout_trajectories (int): Number of trajectories to collect per rollout.

    Returns:
        list[dict[str, Any]]: Collected trajectory data.
    """
    dataset = []

    for timesteps in tqdm.tqdm(range(1, training_steps, rollout_interval), position=0):
        algorithm.learn(total_timesteps=timesteps)

        for trajectory_index in tqdm.tqdm(
            range(rollout_trajectories), position=1, leave=False
        ):
            observations = environment.reset()
            environment_identifiers = [
                str(uuid.uuid4()) for _ in range(environment.num_envs)
            ]

            for rollout_index in tqdm.tqdm(
                range(rollout_steps), position=2, leave=False
            ):
                actions, next_hidden_states = algorithm.predict(observations)
                observations, rewards, done, _ = environment.step(actions)

                if all(done):
                    break

                for index in range(environment.num_envs):
                    if not done[index]:
                        data = {
                            "action": actions[index],
                            "done": done[index],
                            "environment_id": environment_identifiers[index],
                            "image": observations[index].tolist(),
                            "observation": observations[index],
                            "reward": rewards[index],
                            "rollout_index": rollout_index,
                        }

                        dataset.append(data)

    return dataset
