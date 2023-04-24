import torch
from torch import nn
from torch.autograd import Variable
from layers import DenseCapsule, PrimaryCapsule
from layers import SELayer


class CapsuleNet(nn.Module):
    """
    A Capsule Network on Malware Dataset.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.Conv = nn.Sequential(
            nn.Conv2d(input_size[0], 128, 9, 2),         
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.SELayer = SELayer(128, 16)
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(8)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=16*10*10, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(16 * classes, 256, 3, 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256, 128, 3, 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, 2, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 3, 2, 2),
                    nn.BatchNorm2d(3),
                    nn.ReLU(True),
                    
                    nn.Sigmoid()
                )

    def forward(self, x, y=None):
        x = self.Conv(x)
        x = self.SELayer(x)
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        img = (x * y[:, :, None]).view(x.size(0), -1)
        img = torch.randn(x.size(0), 176, 1, 1, device='cuda:0')
        img = self.decoder(img)
        return length, img


