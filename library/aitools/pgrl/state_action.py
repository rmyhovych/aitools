class StateAction(object):
    def __init__(self, obs=None, prob=None, action=None, r=None):
        self.obs = obs
        self.prob = prob
        self.action = action
        self.r = r
