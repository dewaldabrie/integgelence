"""
Provide the Tamagotchi with inputs such as food and see that
the health is affected accordingly.
"""
import unittest
from portal.settings import STATUS_RECEIVER, STATUS_ENCODER_DECODER, INPUT_SENDER, INPUT_ENCODER_DECODER
from tamagotchi.base import Tamagotchi
from portal import Portal
from time import sleep
import numpy as np

class TestInputsToTamagotchi(unittest.TestCase):

    def setUp(self):
        # start tamagotchi in its own process
        self.tamagotchi = Tamagotchi(
            unique_name='Butch',
            time_speedup_factor=5.0,
        )

        # portal need to be run in the same thread as the test
        # in order to properly interact with it
        self.portal = Portal(pet_name='Butch')
        self.status_receiver = STATUS_RECEIVER(
            'Butch_status',
            encoder_decoder=STATUS_ENCODER_DECODER,
        )


    def tearDown(self):
        self.status_receiver.context.destroy(linger=None)
        for socket_bearer in self.tamagotchi._publications.values():
            socket_bearer.context.destroy(linger=None)
        self.portal.sender.context.destroy(linger=None)
        del self.tamagotchi

    @unittest.skip
    def test_neglect_effect_on_health(self):
        """
        Check that health deteriorates with neglect as time passes.
        """
        self.tamagotchi.update()
        status_first = self.status_receiver.receive_message()
        sleep(1)
        self.tamagotchi.update()
        status_second = self.status_receiver.receive_message()

        self.assertGreater(
            status_first['physical_health'],
            status_second['physical_health'],
        )
        self.assertGreater(
            status_first['mental_health'],
            status_second['mental_health'],
        )

    def test_feeding_effect_on_health(self):
        """Check that feeding the Tamagotchi improves it's health"""
        # compose input object
        input = {
            'feed': 1.0, # 1 portion
        }

        # set initial state
        self.tamagotchi.state = np.matrix([
            [-1.],  # fitness
            [-1.],  # nourishment
            [-1.],  # social_stimulation
            [-1.],  # undiseased
        ])

        # send input to Butch the tamagotchi
        self.portal.sender.send_message(input)
        sleep(0.1)
        self.tamagotchi._process_input_queue()

        # check that the social_stimulation state variable has increased
        self.assertGreater(self.tamagotchi.state[1, 0], -1.)

    def test_input_to_state_mapping1(self):
        """Check that the inputs map to reasonable state changes"""
        input = {
            'feed': 1.0,  # 1 portion
        }

        # work with perfect initial state
        self.tamagotchi.state = np.zeros(shape=self.tamagotchi.state.shape, dtype=np.float64)

        # send input via socket and trigger input processing in pet
        self.portal.sender.send_message(input)
        sleep(0.1)

        parsed_input = self.tamagotchi._parse_input(input)
        # check that food is 3rd in alphabetical order of matrix representation of input
        self.assertEqual(parsed_input.T.tolist()[0], [0., 0., 1., 0., 0.])

        self.tamagotchi._process_input_queue()
        # because the state  was set to zeros (perfect), the health was initially optimal
        # feeding more contributes to overfeeding and should push physical health down
        self.assertLess(self.tamagotchi.physical_health, 1.0)

    def test_input_to_state_mapping2(self):
        """
        Check that the inputs map to reasonable state changes
        The act of 'petting' the pet should only improve its mental health, not physical
        """
        input = {
            'pet': 1.0,  # 1 portion
        }

        self.tamagotchi.state = np.matrix([
            [-1.],  # fitness
            [-1.],  # nourishment
            [-1.],  # social_stimulation
            [-1.],  # undiseased
        ])

        parsed_input = self.tamagotchi._parse_input(input)
        self.tamagotchi._process_input_queue()
        # check that pet is 5th in alphabetical order of matrix representation of input
        self.assertEqual(parsed_input.T.tolist()[0], [0., 0., 0., 0., 1.])

        # send input via socket and trigger input processing in pet
        self.portal.sender.send_message(input)
        sleep(0.1)
        self.tamagotchi._process_input_queue()

        # check that the social_stimulation state variable has increased
        self.assertGreater(self.tamagotchi.state[2, 0], -1.)

        #  because the state is set to zeros (perfect), the health should be
        # 1.0 after dotting with the normalized (sum to 1.0) weights
        self.assertGreater(self.tamagotchi.mental_health, 0.0)
        self.assertGreater(self.tamagotchi.mental_health, self.tamagotchi.physical_health)



if __name__ == '__main__':
    unittest.main()