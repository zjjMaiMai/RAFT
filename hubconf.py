"""
Author: zhaohang.zh13@bytedance.com
Date: 2021-03-10 12:01:52
LastEditTime: 2021-03-10 12:11:27
"""
dependencies = ["torch"]

import torch

from core.raft import RAFT as RAFTNet


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