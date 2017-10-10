import os
import time
import pickle
from tamagotchi.domestic import Dog, Cat
from tamagotchi.settings import LOOP_PERIOD
from settings import BASE_DIR
import logging

def run():

    logger = logging.getLogger(__name__)

    # load tamagotchis from file
    instances = load_tamagotchis_from_file(os.path.join(BASE_DIR, 'tamagotchi', 'saved'))

    if instances == []:
        print('No Tamagotchis available, do you want to create one?')
        # TODO implement factory
        instances.append(Dog('Butch'))
    else:
        print('Loaded the following tamagotchis: ', instances)

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
            time.sleep(LOOP_PERIOD - time_delta if time_delta > 0 else 0)
    except KeyboardInterrupt:
        for inst in instances:
            del inst
        print('goodbye.')

def load_tamagotchis_from_file(save_dir):
    file_list = [f for f in os.listdir(save_dir) if f.endswith('.tamag')]
    tamagotchi_list = []
    for f in file_list:
        with open(f, 'r'):
            tamagotchi_list.append(pickle.load(f))
    return tamagotchi_list
