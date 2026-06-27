from .load import load
from .system import system
from .series import series
from . import calibration
from .processing import processing
import ctypes
import os
from pathlib import Path
import sys

try:
    from pyrann_mtp import mtp_bindings
    HAS_MTP = True
except ImportError:
    HAS_MTP = False
# package_dir = Path(__file__).parent
# mtp_exists = any(package_dir.glob('mtp_bindings*.so'))

# loaded_packages = list(sys.modules.keys())
# loaded_packages = [i for i in loaded_packages if i.startswith('pyrann')]
# print(f'loaded_packages = \n{loaded_packages}')

if HAS_MTP:
    # pass
    # from . import mtp_bindings
    __all__ = ['load', 'system', 'series', 'calibration', 'processing', 'mtp_bindings']
else:
    __all__ = ['load', 'system', 'series', 'calibration', 'processing']
# __all__ = ['load', 'system', 'sries', 'calibration', 'processing']

PACKAGE_VERSION = '1.0.0.a1'
print(f'Initializing pyrann version {PACKAGE_VERSION}')
