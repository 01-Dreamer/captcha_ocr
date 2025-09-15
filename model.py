import torch
from torch import nn
import common


class captchaNet(nn.Module):
    def __init__(self):
        super(captchaNet, self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block5 = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=15360, out_features=4096),
          nn.Dropout(0.2),
          nn.ReLU(),
          nn.Linear(in_features=4096, out_features=common.captcha_length*common.captcha_char.__len__())
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

if __name__ == '__main__':
    model = captchaNet()
    x = model(torch.ones(64,1,60,160))
    print(x.shape)
