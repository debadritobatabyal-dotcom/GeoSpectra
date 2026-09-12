import os
import sys
import importlib.util

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

spec = importlib.util.spec_from_file_location("predict", os.path.join(ROOT_DIR, "predict.py"))
predict_module = importlib.util.module_from_spec(spec)
sys.modules["predict"] = predict_module
spec.loader.exec_module(predict_module)
