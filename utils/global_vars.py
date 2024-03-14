# Global Variables

def init():
    global _global_dict
    _global_dict = {}

def set(key, value):
    _global_dict[key] = value

def get(key, default=None):
    try:
        return _global_dict[key]
    except KeyError:
        return default
