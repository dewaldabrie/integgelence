"""
Provide the Tamagotchi with inputs such as food and see that
the health is affected accordingly.
"""
import unittest
from portal.settings import STATUS_RECEIVER
from tamagotchi.run import run as run_tamagotchi
from portal import Portal
from multiprocessing import Process
from time import sleep


class TestInputsToTamagotchi(unittest.TestCase):

    def setUp(self):

        # start tamagotchi in its own process
        self.tamagotchi = Process(
            target=run_tamagotchi,
            args=('Butch',),
            kwargs={'time_speedup_factor':5.},
        )
        self.tamagotchi.start()

        # portal need to be run in the same thread as the test
        # in order to properly interact with it
        self.portal = Portal(pet_name='Butch')
        self.status_receiver = STATUS_RECEIVER('Butch_status')

    def tearDown(self):
        self.status_receiver.context.destroy(linger=None)
        self.tamagotchi.terminate()

    def test_neglect_effect_on_health(self):
        """
        Check that health deteriorates with neglect as time passes.
        """
        status_first = self.status_receiver.receive_message()
        sleep(1)
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
        # get status before
        status_before = self.status_receiver.receive_message()
        # send input to Butch the tamagotchi
        self.portal.sender.send_message(input)
        # get status after
        status_after = self.status_receiver.receive_message()

        # check that physical health improved with food
        self.assertGreater(
            status_after['physical_health'],
            status_before['physical_health'],
        )

if __name__ == '__main__':
    unittest.main()