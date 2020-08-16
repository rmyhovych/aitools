from .abs_gym import AbsGym


class Cartpole(AbsGym):
    def __init__(self):
        super().__init__("CartPole-v1")
