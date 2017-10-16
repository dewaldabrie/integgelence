import os
from settings import BASE_DIR
from tamagotchi.state_encode import PickleEncodeDecode, NoEncodeDecode, JsonEncodeDecode
from common.message_passing import ZMQPublisher, ZMQPairClient

LOOP_PERIOD = 1  # seconds
TAMAGOTCHI_SAVE_DIR = os.path.join(BASE_DIR, 'tamagotchi', 'saved')
STATUS_ENCODER_DECODER = JsonEncodeDecode
STATUS_SENDER = ZMQPublisher
INPUT_RECEIVER = ZMQPairClient
INPUT_ENCODER_DECODER = JsonEncodeDecode