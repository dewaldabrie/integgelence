import unittest
import numpy as np
from tamagotchi.base import Tamagotchi, ALLOWABLE_INPUTS
from common.message_passing import ZMQPairServer
from tamagotchi.state_encode import JsonEncodeDecode
from settings import PET_PORT_MAP
from time import sleep

class TestTamagotchiBase(unittest.TestCase):
    """Unit tests fort the Tamagotchi base class"""
    def setUp(self):
        self.t = Tamagotchi('Butch')
        self.input_sender = ZMQPairServer(
            encoder_decoder=JsonEncodeDecode,
            port=PET_PORT_MAP['Butch']
        )

    def tearDown(self):
        for socket_bearer in self.t._publications.values():
            socket_bearer.context.destroy(linger=None)
        del self.t
        self.input_sender.context.destroy(linger=None)

    def test_state_change_calc(self):
        """
        If an input array is passed to the Tamagotchi, does it update it's state
        correctly?
        """
        # create mock input - supply every possible need
        fake_input = {
            'feed': 1.0,
            'pet': 1.0,
            'excercise': 1.0,
            'immunize': 1.0,
            'clean': 1.0,
        }

        # set state to average before
        self.t.state = np.zeros(len(self.t.state), dtype=np.float64)

        # send the message
        self.input_sender.send_message(fake_input)
        sleep(0.1) # allow for message propogation

        # calculate state change based on fake input
        self.t._process_input_queue()

        self.assertTrue((self.t.state == np.ones(4, dtype=np.float64)).all())


    def test_parse_input(self):
        """
        Does _parse_input preserve the order of ALLOWED_INPUTS?
        """
        input_dict = {
            'feed': 1.0,
            'pet': 1.0,
            'excercise': 1.0,
            'immunize': 1.0,
            'clean': 1.0,
        }

        parsed_input = self.t._parse_input(input_dict)

        self.assertEqual(parsed_input.shape, (len(ALLOWABLE_INPUTS), 1))

    def test_normalize(self):
        """
        Can a vector be normalized to sum to 1.0?
        """
        input = np.array([1,1,1])
        self.assertFalse(input.sum() == 1.0)

        res = self.t._normalize(input)
        self.assertTrue(res.sum() == 1.0, msg=f"Answer is {res}, instead of 1.0")

    def test_normalize_on_zeros(self):
        """
        If a zero vector is normalized, does it sum to 0.0?
        """
        input = np.zeros(5)
        self.assertAlmostEqual(input.sum(), 0.0, places=4)

        res = self.t._normalize(input)
        self.assertTrue(res.sum() == 0.0, msg=f"Answer is {res}, instead of 0.0")

    def test_normalize_on_negatives(self):
        """
        If a zero vector contains negatives , does it sum to 1.0?
        """
        input = np.zeros(5) * -1
        self.assertFalse(input.sum() == 1.0)

        res = self.t._normalize(input)
        self.assertTrue(res.sum() == 0.0, msg=f"Answer is {res}, instead of 0.0")

    def test_health_calcs(self):
        """
        Can mental health be calculated from the state and weights vector?
        """
        # create perfect state (zero)
        self.t.state = np.zeros(len(self.t.state), dtype=np.float64)
        # since weights sum to one, and state is perfect, health should be perfect
        mental_health = self.t.mental_health
        physical_health = self.t.physical_health

        # perfect health is unity
        self.assertAlmostEqual(mental_health, 1.0)
        self.assertAlmostEqual(physical_health, 1.0)

    def test_internal_health_calc(self):
        # create 0.5 state
        self.t.state = np.ones(len(self.t.state), dtype=np.float64)*0.5
        # create unity weights
        weights = np.ones(len(self.t.state), dtype=np.float64)

        self.t._calc_health(weights)

        self.assertEqual(self.t._calc_health(weights), 0.5)

        # create 0.5 state
        self.t.state = np.ones(len(self.t.state), dtype=np.float64) * 0.5
        # create 0.5 weights
        # Note: Because the weights only indicate relative importance,
        # these will be normalized to sum to 1, so is the same as a
        # unity vector.
        # Therefore the result should be the same
        weights = np.ones(len(self.t.state), dtype=np.float64) * 0.5

        self.t._calc_health(weights)

        self.assertEqual(self.t._calc_health(weights), 0.5)

if __name__ == '__main__':
    unittest.main()