import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from video_dataloader import customDataset, transform
from torch.utils.data import DataLoader
import time


class SIAMESE_no_share(nn.Module):
    def __init__(self):
        super(SIAMESE_no_share, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=32,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.avg1_4 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1)
        )
        self.max1_4 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1)
        )
        self.attention_conv1_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=(1, 1),
                stride=1,
            )
        )
        self.conv1_5 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv1_6 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(
                in_channels=256,
                out_channels=1,
                kernel_size=(1, 1, 1),
                stride=1,
                padding=0,
            )
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=32,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv2_4 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.avg2_4 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1)
        )
        self.max2_4 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1)
        )
        self.attention_conv2_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=(1, 1),
                stride=1,
            )
        )
        self.conv2_5 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv2_6 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(
                in_channels=256,
                out_channels=1,
                kernel_size=(1, 1, 1),
                stride=1,
                padding=0,
            )
        )

    def conv1(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        avg = self.avg1_4(x)
        avg = torch.mean(avg, dim=1)
        max = self.max1_4(x)
        max = torch.mean(max, dim=1)
        c = torch.cat([avg, max], dim=1)
        att = self.attention_conv1_4(c)
        att = torch.unsqueeze(att, dim=4)
        mean = att
        nn = x.size(4)
        for n in range(1, nn):
            mean = torch.cat([mean, att], dim=4)
        x = x * mean
        x = self.conv1_5(x)
        x = self.conv1_6(x)
        return x

    def conv2(self, x):
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        avg = self.avg2_4(x)
        avg = torch.mean(avg, dim=1)
        max = self.max2_4(x)
        max = torch.mean(max, dim=1)
        c = torch.cat([avg, max], dim=1)
        att = self.attention_conv2_4(c)
        att = torch.unsqueeze(att, dim=4)
        mean = att
        nn = x.size(4)

        for n in range(1, nn):
            mean = torch.cat([mean, att], dim=4)
        x = x * mean
        x = self.conv2_5(x)
        x = self.conv2_6(x)
        return x

    def forward(self, a, b):
        mean = torch.mean(a, dim=3)
        a = self.conv1(a)
        b = self.conv2(b)
        output = a + b
        return output


def pearson_correlation(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return 1 - cost
