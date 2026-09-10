import os
import sys
import importlib.util

# Ensure root directory is on sys.path for test imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A legacy nested ``project/`` copy also contains predict.py.  Put the active
# repository root first even when it was already added later by a plugin.
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

# Some test runners preload the nested legacy module under the top-level name.
# Bind the test import explicitly to the active root implementation.
spec = importlib.util.spec_from_file_location("predict", os.path.join(ROOT_DIR, "predict.py"))
predict_module = importlib.util.module_from_spec(spec)
sys.modules["predict"] = predict_module
spec.loader.exec_module(predict_module)
