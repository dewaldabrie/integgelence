import unittest
from multiprocessing import Process, Pipe
from common.message_passing import ZMQPublisher, ZMQSubscriber
from common.message_passing import ZMQPairClient, ZMQPairServer
from tamagotchi.state_encode import JsonEncodeDecode
from time import sleep

@unittest.skip('make compatible with new api')
class TestZMQPair(unittest.TestCase):
    """
    Test zmq pair message passing
    """
    @staticmethod
    def pair_one():
        server = ZMQPairServer(encoder_decoder=JsonEncodeDecode)
        for i in range(5):
            server.send_message('hello world')

    @staticmethod
    def pair_two(pipe_conn):
        client = ZMQPairClient(encoder_decoder=JsonEncodeDecode)
        message = client.receive_message()
        pipe_conn.send(message)
        pipe_conn.close()

    def test_message_passing(self):
        pub_proc =  Process(
            target=TestZMQPair.pair_one,
        )

        parent_conn, child_conn = Pipe(duplex=False)
        sub_proc = Process(
            target=TestZMQPair.pair_two,
            args=(child_conn,)
        )

        pub_proc.start()
        sub_proc.start()
        msg = parent_conn.recv()
        pub_proc.terminate()
        sub_proc.terminate()

        self.assertEqual(msg, 'hello world')

    @staticmethod
    def pair_two_coroutine(pipe_conn):
        """Send messages through pipe as they come in"""
        client = ZMQPairClient(encoder_decoder=JsonEncodeDecode)
        message = client.receive_messages()
        pipe_conn.send(message)

    @unittest.skip
    def test_corouting_recieve(self):
        """Test corouting based non-blocking receiving of many messages."""

        pub_proc =  Process(
            target=TestZMQPair.pair_one,
        )

        parent_conn, child_conn = Pipe(duplex=False)
        sub_proc = Process(
            target=TestZMQPair.pair_two_coroutine,
            args=(child_conn,)
        )

        pub_proc.start()
        sub_proc.start()
        msgs = []
        for i in range(5):
            msg = parent_conn.recv()
            self.assertEqual(msg, 'hello world')

        pub_proc.terminate()
        sub_proc.terminate()


@unittest.skip('make compatible with new api')
class TestZMQPubSub(unittest.TestCase):
    """
    Test zmq pub-sub message passing.
    Because the publisher and subscriber has to work independently,
    two seperate processes are used.
    """
    @staticmethod
    def publisher():
        pub = ZMQPublisher(
            'Butch_status',
            JsonEncodeDecode,
            '2000'
        )
        for i in range(5):
            pub.send_message('testing')
            sleep(1)

    @staticmethod
    def subscriber(child_conn):
        sub = ZMQSubscriber(
            'Butch_status',
            JsonEncodeDecode,
            '127.0.0.1',
            '2000')
        msg = sub.receive_message()
        child_conn.send(msg)
        child_conn.close()

    def test_message_passing(self):
        pub_proc =  Process(
            target=TestZMQPubSub.publisher,
        )

        parent_conn, child_conn = Pipe(duplex=False)
        sub_proc = Process(
            target=TestZMQPubSub.subscriber,
            args=(child_conn,)
        )

        pub_proc.start()
        sub_proc.start()
        msg = parent_conn.recv()
        pub_proc.terminate()
        sub_proc.terminate()

        self.assertEqual(msg, 'testing')

