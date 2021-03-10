"""
Author: zhaohang.zh13@bytedance.com
Date: 2021-03-10 12:01:52
LastEditTime: 2021-03-10 12:14:15
"""
dependencies = ["torch"]

import torch
import sys
sys.path.append('core')
from raft import RAFT as RAFTNet


def RAFT(pretrained=True, **kwargs):
    class _ARGS:
        def __init__(self):
            self.small = False
            self.mixed_precision = False
            self.alternate_corr = False

    model = RAFTNet(_ARGS())

    if pretrained:
        checkpoint = (
            "https://github.com/zjjMaiMai/RAFT/releases/download/v1.1/chairs+things.pth"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model