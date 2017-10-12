import sys
import abc
import zmq
import asyncio
import logging
from typing import Union, Callable

logger = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

class Sender(metaclass=abc.ABCMeta):
    """Define common interface for message senders"""
    @abc.abstractmethod
    def send_message(self, mesage):
        pass

class Receiver(metaclass=abc.ABCMeta):
    """Define common interface for message receivers"""
    @abc.abstractmethod
    def receive_message(self):
        pass


class ZMQPublisher(Sender):

    def __init__(self,
                 topic:str,           # topic to publish under (should be unique)
                 port:str='2000',     # port to publish to
                 protocol:str='tcp',  # options: 'tcp', 'inproc', etc.
        ):
        self.topic = topic
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.protocol = protocol
        url = "%s://127.0.0.1:%s" % (self.protocol, self.port)
        self.socket.bind(url)
        logger.info("Publisher bound to %s." % url)

    def send_message(self, message:str):
        """
        Send message using zmq publisher-subscriber pattern
        http://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
        """
        self.socket.send_string(' '.join((self.topic, message)))


class ZMQSubscriber(Receiver):
    """
    Receive status messages from using zmq publisher-subscriber pattern
    http://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
    """

    def __init__(self,
                 topic:str,                 # topic to subscribe to
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
        self.topic = topic
        self.protocol = protocol

        # subscribe
        self.logger = logging.getLogger(__name__)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        url = "%s://%s:%s" % (protocol, host, port)
        self.socket.connect(url)
        logger.info("Subscriber bound to %s." % url)
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic.encode('utf-8'))
        # self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))

    def receive_message(self):
        """Check for a new status message"""
        string = self.socket.recv_string()
        return string.replace(self.topic, '').strip()

    async def receive_messages(self, callback:Callable):
        """Use a coroutine to send new messages to a callback function"""
        # TODO: test this
        while True:
            callback(await self.receive_message())
