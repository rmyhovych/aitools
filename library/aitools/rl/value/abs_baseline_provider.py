class AbsBaselineProvider(object):
    def __call__(self, x):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
