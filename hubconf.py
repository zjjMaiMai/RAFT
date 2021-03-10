dependencies = ["torch"]

import numpy as np
import torch
import torch.nn as nn
import argparse
from core.raft import RAFT as RAFTNet


def RAFT(pretrained=True, checkpoint_name="raft-things.pth", **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args([])
    model = RAFTNet(args)

    if pretrained:
        checkpoint = (
            "https://github.com/zjjMaiMai/RAFT/releases/download/v1.2/{}".format(
                checkpoint_name
            )
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
        )

        class WrappedModel(nn.Module):
            def __init__(self):
                super(WrappedModel, self).__init__()
                self.module = model

            def forward(self, x):
                return self.module(x)

        warpped = WrappedModel()
        warpped.load_state_dict(state_dict)

        return warpped.module
    return model


def raft_forward_helper():
    def pad_x8(img):
        h, w, c = img.shape
        h_8 = h % 8
        w_8 = w % 8
        if h_8 == 0 and w_8 == 0:
            return img, None

        top_pad = h_8 // 2
        down_pad = h_8 - top_pad
        left_pad = w_8 // 2
        right_pad = w_8 - left_pad

        img = np.pad(img, ((top_pad, down_pad), (left_pad, right_pad), (0, 0)), "edge")
        pad_param = top_pad, down_pad, left_pad, right_pad
        return img, pad_param

    def unpad_x8(img, pad_param):
        if pad_param is None:
            return img

        top_pad, down_pad, left_pad, right_pad = pad_param
        return img[top_pad:-down_pad, left_pad:-right_pad, :].copy()

    def _raft_forward_helper_impl(img0, img1, model):
        device = next(model.parameters()).device
        model.eval()

        img0, pad_param = pad_x8(img0)
        img1, pad_param = pad_x8(img1)

        with torch.no_grad():
            img0 = (
                torch.from_numpy(img0).to(device).permute(2, 0, 1).float().unsqueeze(0)
            )
            img1 = (
                torch.from_numpy(img1).to(device).permute(2, 0, 1).float().unsqueeze(0)
            )

            _, flow_up = model(img0, img1, iters=20, test_mode=True)

        flow_up = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
        flow_up = unpad_x8(flow_up, pad_param)

        return flow_up

    return _raft_forward_helper_impl