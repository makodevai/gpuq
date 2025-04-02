from enum import IntFlag, auto


class Provider(IntFlag):
    ANY = 0

    CUDA = auto()
    HIP = auto()

    ALL = CUDA|HIP
