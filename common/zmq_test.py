import unittest
from multiprocessing import Process, Pipe
from common.message_passing import ZMQPublisher, ZMQSubscriber
from common.message_passing import ZMQPairClient, ZMQPairServer
from tamagotchi.state_encode import JsonEncodeDecode
from time import sleep

class TestZMQPair(unittest.TestCase):
    """
    Test zmq pair message passing
    """

    def test_message_passing(self):
        server = ZMQPairServer(encoder_decoder=JsonEncodeDecode)
        client = ZMQPairClient(encoder_decoder=JsonEncodeDecode)

        server.send_message('hello world')
        msg = client.receive_message()

        self.assertEqual(msg, 'hello world')



@unittest.skip('make compatible with new api')
class TestZMQPubSub(unittest.TestCase):
    """
    Test zmq pub-sub message passing.
    Because the publisher and subscriber has to work independently,
    two seperate processes are used.
    """
    def test_message_passing(self):

        publisher = ZMQPublisher(
            'Butch_status',
            JsonEncodeDecode,
            '2000'
        )
        subscriber = ZMQSubscriber(
            'Butch_status',
            JsonEncodeDecode,
            '127.0.0.1',
            '2000'
        )

        publisher.send_message('testing')
        msg = subscriber.receive_message()

        self.assertEqual(msg, 'testing')

