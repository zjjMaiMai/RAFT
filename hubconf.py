"""
Author: zhaohang.zh13@bytedance.com
Date: 2021-03-10 12:01:52
LastEditTime: 2021-03-10 12:21:44
"""
dependencies = ["torch"]

import torch
import argparse
from core.raft import RAFT as RAFTNet


def RAFT(pretrained=True, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args([])
    model = RAFTNet(args)

    if pretrained:
        checkpoint = (
            "https://github.com/zjjMaiMai/RAFT/releases/download/v1.1/chairs+things.pth"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model