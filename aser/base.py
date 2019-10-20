import json

class JsonSerializedObject(object):
    def __init__(self):
        pass

    def to_dict(self):
        return self.__dict__

    def from_dict(self, d):
        for attr_name in self.__dict__:
            self.__setattr__(attr_name, d[attr_name])

    def encode(self, encoding="utf-8"):
        if encoding == "utf-8":
            msg = json.dumps(self.__dict__).encode("utf-8")
        elif encoding == "ascii":
            msg = json.dumps(self.__dict__).encode("ascii")
        else:
            msg = self.__dict__
        return msg

    def decode(self, msg, encoding="utf-8"):
        if encoding == "utf-8":
            decoded_dict = json.loads(msg.decode("utf-8"))
        elif encoding == "ascii":
            decoded_dict = json.loads(msg.decode("ascii"))
        else:
            decoded_dict = msg
        self.from_dict(decoded_dict)
        return self

