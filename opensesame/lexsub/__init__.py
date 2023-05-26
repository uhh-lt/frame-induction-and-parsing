import os
import sys
from pathlib import Path

try:
    MODULE = os.path.dirname(os.path.realpath(__file__))

    if not os.path.dirname(Path(MODULE)) in sys.path:
        sys.path.insert(0, os.path.dirname(Path(MODULE)))

    if not os.path.dirname(Path(MODULE).parent) in sys.path:
        sys.path.insert(0, os.path.dirname(Path(MODULE).parent))

#     if not os.path.dirname(Path(MODULE).parent.parent) in sys.path:
#         sys.path.insert(0, os.path.dirname(Path(MODULE).parent.parent))
        
except:
    pass

# try:
#     MODULE = os.path.dirname(os.path.realpath(__file__))
#     if not os.path.join(MODULE, "..") in sys.path:
#         sys.path.insert(0, os.path.join(MODULE, ".."))
#     if not os.path.join(MODULE, "..", "..") in sys.path:
#         sys.path.insert(0, os.path.join(MODULE, "..", ".."))
# except:
#     pass