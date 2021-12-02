def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d
