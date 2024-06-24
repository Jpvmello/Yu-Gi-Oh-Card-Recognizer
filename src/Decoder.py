import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unflatten = nn.Unflatten(1, (256, 20, 14))
        self.block1 = self._gen_block(in_channels = 256, out_channels = 128)
        self.block2 = self._gen_block(in_channels = 128, out_channels =  64)
        self.block3 = self._gen_block(in_channels =  64, out_channels =  32)
        self.block4 = self._gen_block(in_channels =  32, out_channels =  16)
        self.block5 = self._gen_block(in_channels =  16, out_channels =   3)
    
    def _gen_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.unflatten(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x