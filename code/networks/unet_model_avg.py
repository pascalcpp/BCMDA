""" Full assembly of the parts to form the complete network """
from .unet_parts import *
class StochasticClassifier(nn.Module):
    def __init__(self, num_features, num_classes, temp=0.05):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.temp = temp

    def forward(self, x):
        weight = self.mu
        weight_norm = F.normalize(weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        score = torch.einsum('bchw,nc->bnhw', x_norm, weight_norm)
        score = score / self.temp
        return score

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, temp=0.05, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.temp = temp

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc1_sim = StochasticClassifier(64, n_classes, temp)
        self.outc2_sim = StochasticClassifier(64, n_classes, temp)
        self.outc = OutConv(64, n_classes)

    def classify_sim_avg(self, x):
        weight = (self.outc1_sim.mu + self.outc2_sim.mu) / 2.0
        weight_norm = F.normalize(weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        score = torch.einsum('bchw,nc->bnhw', x_norm, weight_norm)
        score = score / self.temp
        return score


    def classify_linear(self, x):
        score = self.outc(x)
        return score

    def classify_sim1(self, x, ratio=0.0):
        weight = (self.outc1_sim.mu * 2.0 + self.outc2_sim.mu * (1.0 + ratio)) / (3.0 + ratio)
        weight_norm = F.normalize(weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        score = torch.einsum('bchw,nc->bnhw', x_norm, weight_norm)
        score = score / self.temp
        return score

    def classify_sim2(self, x, ratio=0.0):
        weight = (self.outc2_sim.mu * 2.0 + self.outc1_sim.mu * (1.0 + ratio)) / (3.0 + ratio)
        weight_norm = F.normalize(weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        score = torch.einsum('bchw,nc->bnhw', x_norm, weight_norm)
        score = score / self.temp
        return score

    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return [x1, x2, x3, x4, x5]

    def seg_decoder(self, feature_list):
        x1, x2, x3, x4, x5 = feature_list
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def forward(self, x):
        dict_ret = {}
        feature_list = self.encoder(x)
        seg_last_fts = self.seg_decoder(feature_list)
        dict_ret['last_fts'] = seg_last_fts

        return dict_ret