import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSamplingBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)  # up path
        x_p = self.x_conv(x_p)  # cross path
        cat_p = torch.cat([up_p, x_p], dim=1)  # dim 1 means concat from means left to right
        return self.bn(F.relu(cat_p))


# in order to pass output of resnet layers to upsampling blocks
class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inp, out):
        self.features = out

    def remove(self):
        self.hook.remove()


# inspired by Jeremy Howards idea on using resnet in U-net
def get_base():
    f = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
    base_model = nn.Sequential(*list(f.children())[:-2])
    return base_model


class SegmenterModel(nn.Module):
    def __init__(self):
        super(SegmenterModel, self).__init__()

        self.resnet = get_base()
        self.sfs = [SaveFeatures(self.resnet[i]) for i in [2, 4, 5, 6]]

        self.up1 = UpSamplingBlock(512, 256, 256)
        self.up2 = UpSamplingBlock(256, 128, 256)
        self.up3 = UpSamplingBlock(256, 64, 256)
        self.up4 = UpSamplingBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        # resnet34 as downsampling and bottleneck
        x = F.relu(self.resnet(x))
        # upsampling
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x

    def close(self):
        for sf in self.sfs:
            sf.remove()
