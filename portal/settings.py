from tamagotchi.state_encode import PickleEncodeDecode, NoEncodeDecode, JsonEncodeDecode
from common.message_passing import ZMQSubscriber, ZMQPairServer

LOOP_PERIOD = 1  # seconds
STATUS_ENCODER_DECODER = JsonEncodeDecode
STATUS_RECEIVER = ZMQSubscriber
INPUT_SENDER = ZMQPairServer
INPUT_ENCODER_DECODER = JsonEncodeDecode
