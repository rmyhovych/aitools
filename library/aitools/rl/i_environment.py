class IEnvironment(object):
    def reset(self):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def getSizeObs(self):
        raise NotImplementedError

    def getSizeAction(self):
        raise NotImplementedError
