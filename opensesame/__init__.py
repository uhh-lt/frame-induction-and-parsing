import os
import sys
from pathlib import Path

try:
    MODULE = os.path.dirname(os.path.realpath(__file__))

    if not os.path.dirname(Path(MODULE)) in sys.path:
        sys.path.insert(0, os.path.dirname(Path(MODULE)))
        
except:
    pass

