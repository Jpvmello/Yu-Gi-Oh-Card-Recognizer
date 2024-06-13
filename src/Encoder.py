import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self._gen_block(in_channels =   3, out_channels =  16)
        self.block2 = self._gen_block(in_channels =  16, out_channels =  32)
        self.block3 = self._gen_block(in_channels =  32, out_channels =  64)
        self.block4 = self._gen_block(in_channels =  64, out_channels = 128)
        self.block5 = self._gen_block(in_channels = 128, out_channels = 256)
    
    def _gen_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x