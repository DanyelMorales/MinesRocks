from enum import Enum


class Strategy(Enum):
    ENCODE_BY_CLASSES = 1
    TAKE_LABEL_AS_INDEX = 2
    REPLACE_LABEL_BY_VALUE = 3
    CUSTOM_MAPPING = 4


