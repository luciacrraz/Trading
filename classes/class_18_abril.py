from typing import Tuple

import gymnasium
import numpy as np
import matplotlib.pyplot as plt


class RandomStateAgent:
    def __init__(self, env_name: str, epochs: int = 10, trials=10, **kwargs):
        self.trials = trials
        self.env_name = env_name
        self.epochs = epochs

        # create a gym environment
        self.env = gymnasium.make(self.env_name, **kwargs)

    def run(self) -> tuple[int, None]:
        # for each epoch, restart env, RL loop
        for trial in range(self.trials):
            epoch_rewards = []
            best_reward = 0
            best_weights = None

            # set random weights
            ...
            for epoch in range(self.epochs):
                done = False
                state_0, _ = self.env.reset()

                episode_reward = 0
                while not done:
                    # choose a random action
                    a = self.env.action_space.sample()

                    # take action a
                    state_1, reward, done, trunc, info = self.env.step(a)

                    episode_reward += reward

                    state_0 = state_1
                epoch_rewards.append(episode_reward)
        self.env.close()
        return epoch_rewards, best_reward, best_weights


if __name__ == "__main__":
    agent = RandomStateAgent(env_name="CartPole-v0", epochs=10, render_mode="human")
    history = agent.run()

    plt.plot(history)
    plt.show()
