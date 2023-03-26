def is_posintstring(s):
    try:
        temp = int(s)
        if temp > 0:
            return True
        else:
            return False
    except ValueError:
        return False

