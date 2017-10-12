"""Receive messages from pets on health, send inputs and actions"""

import sys
import zmq
import logging
import time
from typing import Union
from .settings import STATUS_ENCODER_DECODER, STATUS_RECEIVER, LOOP_PERIOD


def run():
    logger = logging.getLogger(__name__)
    # get messages from Butch
    BUTCH_STATUS = 'Butch_status'  # topic name
    subscriptions = {
        BUTCH_STATUS: STATUS_RECEIVER(BUTCH_STATUS)
    }

    try:
        while True:
            time_start = time.time()
            # get new status
            status = subscriptions[BUTCH_STATUS].receive_message()
            print("New status from Butch: %s" % status)

            # sleep for the remainder of the period
            time_delta = time.time() - time_start
            if time_delta < 0:
                logger.warning(f"Loop period over-run by {time_delta} seconds!")
            time.sleep(LOOP_PERIOD - time_delta if time_delta > 0 else 0)
    except KeyboardInterrupt:
        print('goodbye.')

