class Config(object):
    def __init__(self, d):
        self.recursive_update(d)

    def recursive_update(self, d):
        new_d = {}
        for key, val in d.items():
            if isinstance(val, dict):
                new_val = Config(val)
            else:
                new_val = val
            new_d[key] = new_val
        self.__dict__.update(new_d)