import sys
import abc
import zmq
import asyncio
import logging
from typing import Union, Callable, Iterator, Any

logger = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

class Sender(metaclass=abc.ABCMeta):
    """Define common interface for message senders"""
    @abc.abstractmethod
    def send_message(self, message):
        pass

class Receiver(metaclass=abc.ABCMeta):
    """Define common interface for message receivers"""
    @abc.abstractmethod
    def receive_message(self):
        pass

class ZMQServer:
    def __init__(self,
                 socket_type,         # for exampe zmq.PAIR, zmq.pub
                 port:str='2000',     # port to publish to
                 protocol:str='tcp',  # options: 'tcp', 'inproc', etc.
        ):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(socket_type)
        self.protocol = protocol
        url = "%s://*:%s" % (self.protocol, self.port)
        self.socket.bind(url)
        logger.info("Server bound to %s." % url)

    def __del__(self):
        self.context.destroy(linger=None)


class ZMQClient:
    def __init__(self,
                 socket_type,                 # ex. zmq.PAIR, zmq.SUB
                 host:str='127.0.0.1',      # like 'localhost'
                 port:str='2000',           # port to listen on
                 protocol:str='tcp'         # options: 'tcp', 'inproc', etc.
    ):
        """
        Establish communication with another entity with the purpose
        of receiving the status message of that entity
        """
        self.host = host
        self.port = port
        self.protocol = protocol

        # subscribe
        self.logger = logging.getLogger(__name__)
        self.context = zmq.Context()
        self.socket = self.context.socket(socket_type)
        url = "%s://%s:%s" % (protocol, host, port)
        self.socket.connect(url)
        logger.info("Client bound to %s." % url)

    def __del__(self):
        self.context.destroy(linger=None)


class ZMQPairServer(Receiver, Sender, ZMQServer):
    """
    Two way communication between two entities.
    Server and client interface it identical.
    Initialization sequence is slightly different.
    """
    def __init__(self,
                 encoder_decoder=None,
                 port:str='2000',     # port to publish to
                 protocol:str='tcp',  # options: 'tcp', 'inproc', etc.
        ):
        super().__init__(
            zmq.PAIR,
            port=port,
            protocol=protocol,
        )
        self.encoder_decode = encoder_decoder() if encoder_decoder else None

    def send_message(self, message):
        if self.encoder_decode:
            message = self.encoder_decode.encode(message)
        self.socket.send_string(message)

    def receive_message(self):
        message = self.socket.recv_string()
        if self.encoder_decode:
            message = self.encoder_decode.decode(message)
        return message



class ZMQPairClient(Receiver, Sender, ZMQClient):
    """
    Two way communication between two entities.
    Server and client interface it identical.
    Initialization sequence is slightly different.
    """

    def __init__(self,
                 encoder_decoder=None,
                 host: str = '127.0.0.1',  # like 'localhost'
                 port: str = '2000',  # port to listen on
                 protocol: str = 'tcp'  # options: 'tcp', 'inproc', etc.
                 ):
        super().__init__(
            zmq.PAIR,
            host=host,
            port=port,
            protocol=protocol,
        )
        self.encoder_decode = encoder_decoder() if encoder_decoder else None

    def send_message(self, message):
        if self.encoder_decode:
            message = self.encoder_decode.encode(message)
        self.socket.send_string(message)

    def receive_message(self, flags=0):
        message = self.socket.recv_string(flags)
        if self.encoder_decode:
            message = self.encoder_decode.decode(message)
        return message

    def receive_messages(self):
        """Iteratively yeild all availables messages"""
        while True:
            try:
                yield self.receive_message(zmq.NOBLOCK)
            except zmq.error.ZMQError:
                return


class ZMQPublisher(Sender, ZMQServer):

    def __init__(self,
                 topic:str,            # topic to publish under (should be unique)
                 encoder_decoder=None, # class used for encoding and decoding messages
                 port:str='2000',      # port to publish to
                 protocol:str='tcp',   # options: 'tcp', 'inproc', etc.
        ):
        super().__init__(
            zmq.PUB,
            port=port,
            protocol=protocol
        )
        self.topic = topic
        self.encoder_decode = encoder_decoder() if encoder_decoder else None


    def send_message(self, message:str):
        """
        Send message using zmq publisher-subscriber pattern
        http://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
        """
        if self.encoder_decode:
            message = self.encoder_decode.encode(message)
        self.socket.send_string(' '.join((self.topic, message)))


class ZMQSubscriber(Receiver, ZMQClient):
    """
    Receive status messages from using zmq publisher-subscriber pattern
    http://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
    """

    def __init__(self,
                 topic:str,                 # topic to publish under (should be unique)
                 encoder_decoder=None,      # class used for encoding and decoding messages
                 host:str='127.0.0.1',      # like 'localhost'
                 port:str='2000',           # port to listen on
                 protocol:str='tcp'         # options: 'tcp', 'inproc', etc.
    ):
        """
        Establish communication with another entity with the purpose
        of receiving the status message of that entity
        """
        super().__init__(
            zmq.SUB,
            host=host,
            port=port,
            protocol=protocol,
        )
        self.topic = topic
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic.encode('utf-8'))
        self.encoder_decode = encoder_decoder() if encoder_decoder else None

    def receive_message(self, flags=zmq.NOBLOCK):
        """Check for a new status message"""
        string = None
        # skip to the latest message
        while True:
            try:
                res = self.socket.recv_string(flags=flags)
                string = res
            except zmq.error.ZMQError:
                break
        if string:
            string = string.replace(self.topic, '').strip()
            if self.encoder_decode:
                message = self.encoder_decode.decode(string)
                return message
            return string

    async def receive_messages(self, callback:Callable):
        """Use a coroutine to send new messages to a callback function"""
        # TODO: test this
        while True:
            callback(await self.receive_message())
