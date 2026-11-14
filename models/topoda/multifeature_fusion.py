import torch
import torch.nn as nn
import torch.nn.functional as F
from DCNv4.modules import DCNv4


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class PinwheelShapedConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        assert c1 % 4 == 0 and c2 % 4 == 0, \
            f"PinwheelShapedConv expects channels divisible by 4, got c1={c1}, c2={c2}"

        # (left, right, top, bottom) padding for the four directions
        self.pads = [
            nn.ZeroPad2d(padding=(k, 0, 1, 0)),   # left
            nn.ZeroPad2d(padding=(0, k, 0, 1)),   # right
            nn.ZeroPad2d(padding=(0, 1, k, 0)),   # top
            nn.ZeroPad2d(padding=(1, 0, 0, k)),   # bottom
        ]

        # Each group has c1/4 input channels and produces c2/4 output channels
        self.cw = Conv(c1 // 4, c2 // 4, (1, k), s=s, p=0)   # horizontal kernels
        self.ch = Conv(c1 // 4, c2 // 4, (k, 1), s=s, p=0)   # vertical kernels
        self.fuse = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        # Split input into 4 channel groups
        x_groups = torch.chunk(x, 4, dim=1)
        yw0 = self.cw(self.pads[0](x_groups[0]))   # left
        yw1 = self.cw(self.pads[1](x_groups[1]))   # right
        yh0 = self.ch(self.pads[2](x_groups[2]))   # top
        yh1 = self.ch(self.pads[3](x_groups[3]))   # bottom
        return self.fuse(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class DirectionalFeatureExtraction(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.pinwheel = PinwheelShapedConv(in_chan, out_chan)
        # Interpolating a learned spatial grid supports variable image sizes.
        self.gamma = nn.Parameter(torch.ones(1, out_chan, 8, 8))

    def forward(self, x):
        f_dir = self.pinwheel(x)
        # Dynamic spatial-channel re-weighting
        gamma = F.interpolate(self.gamma, size=f_dir.shape[-2:], mode='bilinear', align_corners=False)
        f_aligned = f_dir * gamma
        return f_aligned


class DeformableFeatureAlignment(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_conv = DirectionalFeatureExtraction(in_channels, out_channels)
        # Offset prediction sees both scales; sampled values come from the
        # upsampled lower-resolution map.
        self.offset_context = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.dcn = DCNv4(
            channels=out_channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=8,
        )
        self.relu = nn.ReLU(inplace=True)
        self.adjust = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, feat_high_res, feat_low_res):
        """
        Args:
            feat_high_res: higher-resolution reference feature E_i (B, C, H, W)
            feat_low_res:  lower-resolution feature E_{i-1} (B, C, H', W')
        Returns:
            Aligned and fused feature map (B, C, H, W)
        """
        # Upsample lower-resolution feature to match high-resolution size
        if feat_low_res.size()[2:] != feat_high_res.size()[2:]:
            feat_low_res_up = F.interpolate(
                feat_low_res, size=feat_high_res.size()[2:],
                mode='bilinear', align_corners=False
            )
        else:
            feat_low_res_up = feat_low_res

        # Lateral connection for the high-resolution reference
        feat_arm = self.lateral_conv(feat_high_res)

        # Joint context predicts offsets; values remain on the lower-res branch.
        context = self.offset_context(torch.cat([feat_low_res_up, feat_arm], dim=1))
        feat_align = self.relu(self.adjust(self.dcn(feat_low_res_up, offset_context=context)))

        return feat_align + feat_arm


class LocallyAwareFeatureAlignment(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        self.align_modules = nn.ModuleList()
        for in_channels in in_channels_list[:-1]:
            self.align_modules.append(
                DeformableFeatureAlignment(in_channels, out_channels)
            )

        # Transform the last (coarsest) layer
        self.last_conv = DirectionalFeatureExtraction(in_channels_list[-1], out_channels)

    def forward(self, features):
        assert len(features) == len(self.align_modules) + 1

        results = []
        # Process highest level (coarsest) features
        prev_features = self.last_conv(features[-1])
        results.append(prev_features)

        # Top-down pathway: from coarse to fine
        for feat, align_module in zip(
                reversed(features[:-1]),
                reversed(self.align_modules)):
            prev_features = align_module(feat, prev_features)
            results.insert(0, prev_features)

        return results
