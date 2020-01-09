import random

# Окраска информационного текста
def INFOcolored(text):
    return (f"\x1b[1;33m{text}\x1b[0m")

# Цвет ошибки
def Errorcolored(text):
    return (f"\x1b[0;31m{text}\x1b[0m")

def SimpleTextColor(text):
    return (f"\x1b[0;44m{text}\x1b[0m")



def GenRGB():
    r = lambda: random.randint(0, 255)
    return r(), r(), r()
