try:
    from .segmenteverygrain import *
except ModuleNotFoundError:
    # Lightweight experiment utilities can be imported without optional ML/data
    # dependencies installed. Core training/prediction functions still require
    # the full environment and will raise when imported directly.
    pass
