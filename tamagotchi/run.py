import os
import time
import pickle
from tamagotchi.domestic import Tamagotchi
from tamagotchi.settings import LOOP_PERIOD, TAMAGOTCHI_SAVE_DIR
from settings import BASE_DIR, PET_PORT_MAP
import logging

def run(pet_name:str, time_speedup_factor:float=1):

    logger = logging.getLogger(__name__)


    instances = []
    # TODO implement factory
    instances.append(Tamagotchi(pet_name, time_speedup_factor=time_speedup_factor))

    try:
        while True:
            time_start = time.time()

            #update all Tamagotchis
            for inst in instances:
                inst.update()

            # sleep for the remainder of the period
            time_delta = time.time() - time_start
            if time_delta < 0:
                logger.warning(f"Loop period over-run by {time_delta} seconds!")
            time.sleep(LOOP_PERIOD - time_delta if time_delta < LOOP_PERIOD else 0)

    except KeyboardInterrupt:
        for inst in instances:
            for socket_bearer in inst._publications.values():
                socket_bearer.context.destroy(linger=None)
        print('goodbye.')
