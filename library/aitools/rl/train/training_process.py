from matplotlib import pyplot as plt
import numpy as np

from .abs_agent_trainer import AbsAgentTrainer
from ..value import ZeroBaselineProvider
from ..i_environment import IEnvironment


class TrainingProcess(object):
    def __init__(self, agent_trainer: AbsAgentTrainer):
        self.agent_trainer = agent_trainer
        self.baseline_provider = ZeroBaselineProvider()
        self.env: IEnvironment = None

        self.to_plot = False

    def set_baseline_provider(self, baseline_provider):
        self.baseline_provider = baseline_provider
        return self

    def set_env(self, env: IEnvironment):
        self.env = env
        return self

    def plot(self):
        self.to_plot = True
        return self

    def train(self, n_trajectories: int, n_episodes: int):
        losses = []
        trajectory_sizes = []
        decisivenesses = []
        try:
            for episode in range(n_episodes):
                average_trajectory_size = 0.0
                for _ in range(n_trajectories):
                    average_trajectory_size += len(
                        self.agent_trainer.collect_trajectory(self.env)
                    )
                decisivenesses.append(self.agent_trainer.get_decisiveness())

                losses.append(self.agent_trainer.train(self.baseline_provider))
                self.baseline_provider.update()

                trajectory_sizes.append(average_trajectory_size / n_trajectories)
                print("{}\t-> {:0.2f}".format(episode, trajectory_sizes[-1]))

        except KeyboardInterrupt:
            pass

        if self.to_plot:
            xs = [x for x in range(len(trajectory_sizes))]
            a, b = np.polyfit(xs, trajectory_sizes, 1)

            fig, axes = plt.subplots(nrows=1, ncols=3)
            axes[0].plot(losses)
            axes[1].plot(xs, trajectory_sizes, "g")
            axes[1].plot(xs, [a * x + b for x in xs], "r")
            axes[2].plot(decisivenesses)
            axes[2].set_ylim([0.0, 1.0])

            fig.tight_layout()
            plt.show()

        return (losses, trajectory_sizes, decisivenesses)
