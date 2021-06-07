import json


class JsonSerializedObject(object):
    """ Object that supports json serialization

    """
    def __init__(self):
        pass

    def to_dict(self, **kw):
        """ Convert a object to a dictionary

        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the converted dictionary that contains necessary information
        :rtype: Dict[str, object]
        """

        return dict(self.__dict__) # shadow copy

    def from_dict(self, d, **kw):
        """ Convert a dictionary to an object

        :param d: a dictionary contains necessary information
        :type d: Dict[str, object]
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the converted object
        :rtype: aser.object.JsonSerializedObject
        """

        for attr_name in d:
            self.__setattr__(attr_name, d[attr_name])
        return self

    def encode(self, encoding="utf-8", **kw):
        """ Encode the object

        :param encoding: the encoding format
        :type encoding: str (default = "utf-8")
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the encoded bytes
        :rtype: bytes
        """

        d = self.to_dict(**kw)
        if encoding == "utf-8":
            msg = json.dumps(d).encode("utf-8")
        elif encoding == "ascii":
            msg = json.dumps(d).encode("ascii")
        else:
            msg = d
        return msg

    def decode(self, msg, encoding="utf-8", **kw):
        """ Decode the object

        :param msg: the encoded bytes
        :type msg: bytes
        :param encoding: the encoding format
        :type encoding: str (default = "utf-8")
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the decoded bytes
        :rtype: bytes
        """

        if encoding == "utf-8":
            decoded_dict = json.loads(msg.decode("utf-8"))
        elif encoding == "ascii":
            decoded_dict = json.loads(msg.decode("ascii"))
        else:
            decoded_dict = msg
        self.from_dict(decoded_dict, **kw)
        return self
