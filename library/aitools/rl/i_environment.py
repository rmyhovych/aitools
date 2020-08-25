class IEnvironment(object):
    def reset(self):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def get_size_obs(self):
        raise NotImplementedError

    def get_size_action(self):
        raise NotImplementedError
