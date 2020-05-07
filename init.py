from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.python.util import deprecation

"""
Because of error on Linux:
Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize.
"""
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

"""
Hide deprecation warnings.
"""
deprecation._PRINT_DEPRECATION_WARNINGS = False
