from tamagotchi.state_encode import PickleEncodeDecode, NoEncodeDecode, JsonEncodeDecode
from common.message_passing import ZMQSubscriber

LOOP_PERIOD = 1  # seconds
STATUS_ENCODER_DECODER = JsonEncodeDecode
STATUS_RECEIVER = ZMQSubscriber