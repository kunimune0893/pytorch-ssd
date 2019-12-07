# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024, debug_dk=None):
        super(MobileNetV1, self).__init__()
        
        self.debug_dk = debug_dk

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),      #  0
            conv_dw(32, 64, 1),     #  1
            conv_dw(64, 128, 2),    #  2
            conv_dw(128, 128, 1),   #  3
            conv_dw(128, 256, 2),   #  4
            conv_dw(256, 256, 1),   #  5
            conv_dw(256, 512, 2),   #  6
            conv_dw(512, 512, 1),   #  7
            conv_dw(512, 512, 1),   #  8
            conv_dw(512, 512, 1),   #  9
            conv_dw(512, 512, 1),   # 10
            conv_dw(512, 512, 1),   # 11
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        if self.debug_dk == "dump":
            print( "after model(x), x.shape=", x.shape )
            if len(x.data.cpu().numpy().reshape(-1)) % 16 == 0:
                np.savetxt( "./logs/" + "after_model.csv", x.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
            else:
                np.savetxt( "./logs/" + "after_model.csv", x.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
        x = F.avg_pool2d(x, 7)
        if self.debug_dk == "dump":
            print( "after avg_pool2d(x), x.shape=", x.shape )
            if len(x.data.cpu().numpy().reshape(-1)) % 16 == 0:
                np.savetxt( "./logs/" + "after_avg_pool2d.csv", x.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
            else:
                np.savetxt( "./logs/" + "after_avg_pool2d.csv", x.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
        x = x.view(-1, 1024)
        x = self.fc(x)
        if self.debug_dk == "dump":
            print( "after fc(x), x.shape=", x.shape )
            if len(x.data.cpu().numpy().reshape(-1)) % 16 == 0:
                np.savetxt( "./logs/" + "after_fc.csv", x.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
            else:
                np.savetxt( "./logs/" + "after_fc.csv", x.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
        return x