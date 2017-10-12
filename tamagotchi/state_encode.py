import pickle
import json
import abc

class EncodeDecode(metaclass=abc.ABCMeta):
    """ Base class for encoder/decoder"""
    @abc.abstractmethod
    def encode(self, message):
        """Encode the message"""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, message):
        """Decode the message"""
        raise NotImplementedError

class NoEncodeDecode(EncodeDecode):
    """Encode and decode the message using the pickle library"""

    def encode(self, message):
        return message

    def decode(self, message):
        return message


class PickleEncodeDecode(EncodeDecode):
    """Encode and decode the message using the pickle library"""

    def encode(self, message):
        return pickle.dumps(message)

    def decode(self, message):
        return pickle.loads(message)


class JsonEncodeDecode(EncodeDecode):
    """Encode and decode the message using the pickle library"""

    def encode(self, message):
        return json.dumps(message)

    def decode(self, message):
        return json.loads(message)