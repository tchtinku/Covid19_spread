from covid19_spread.bar import BARCV
import yaml
from argparse import Namespace
import torch as th

class TestBatchedInference:
    def test_batched_inference(self):
        with th.no_grad():
            th.set_default_tensor_type(th.DoubleTensor)
            th.manual_seed(0)
            mod = BARCV()
            cfg = yaml.safe_load(open("cv/us.yml"))
            opt = Namespace(
                **{
                    k: v[0] if isinstance(v, list) else v
                    for k, v in cfg["bar"]["train"].items()
                }
            )