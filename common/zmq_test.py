import unittest
from multiprocessing import Process, Pipe
from common.message_passing import ZMQPublisher, ZMQSubscriber
from time import sleep


class TestZMQPubSub(unittest.TestCase):
    """
    Test zmq message passing.
    Because the publisher and subscriber has to work independently,
    two seperate processes are used.
    """
    @staticmethod
    def publisher():
        pub = ZMQPublisher('Butch_status', '2000')
        for i in range(5):
            pub.send_message('testing')
            sleep(1)

    @staticmethod
    def subscriber(child_conn):
        sub = ZMQSubscriber('Butch_status', '127.0.0.1', '2000')
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

