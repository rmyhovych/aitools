class StateAction(object):
    def __init__(self, obs=None, prob=None, action=None, r=None):
        self.obs = obs
        self.prob = prob
        self.action = action
        self.r = r

    def __str__(self):
        return "[obs={};prob={};action={};r={}]".format(
            self.obs, self.prob, self.action, self.r
        )

    def __repr__(self):
        return str(self)
