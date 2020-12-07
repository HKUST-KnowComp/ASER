import json

class JsonSerializedObject(object):
    def __init__(self):
        pass

    def to_dict(self, **kw):
        return dict(self.__dict__) # shadow copy

    def from_dict(self, d, **kw):
        for attr_name in d:
            self.__setattr__(attr_name, d[attr_name])
        return self

    def encode(self, encoding="utf-8", **kw):
        d = self.to_dict(**kw)
        if encoding == "utf-8":
            msg = json.dumps(d).encode("utf-8")
        elif encoding == "ascii":
            msg = json.dumps(d).encode("ascii")
        else:
            msg = d
        return msg

    def decode(self, msg, encoding="utf-8", **kw):
        if encoding == "utf-8":
            decoded_dict = json.loads(msg.decode("utf-8"))
        elif encoding == "ascii":
            decoded_dict = json.loads(msg.decode("ascii"))
        else:
            decoded_dict = msg
        self.from_dict(decoded_dict, **kw)
        return self

