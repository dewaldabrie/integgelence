"""Receive messages from pets on health, send inputs and actions"""

import sys
import zmq
import logging
import time
from typing import Union
from portal import Portal
from .settings import LOOP_PERIOD

def run():
    logger = logging.getLogger(__name__)


    # create portal (for menu options)
    pet_portal = Portal(pet_name='Butch')

    try:
        while True:
            time_start = time.time()

            # show pet input
            pet_portal.main()


            # sleep for the remainder of the period
            time_delta = time.time() - time_start
            if time_delta < 0:
                logger.warning(f"Loop period over-run by {time_delta} seconds!")
            time.sleep(LOOP_PERIOD - time_delta if time_delta < LOOP_PERIOD else 0)
    except KeyboardInterrupt:
        for socket_bearer in pet_portal.subscriptions.values():
            socket_bearer.context.destroy(linger=None)
        print('goodbye.')

