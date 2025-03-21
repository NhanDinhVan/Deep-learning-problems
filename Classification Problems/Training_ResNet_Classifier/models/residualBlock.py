import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, expansion, is_bottle_neck, stride):

        """
        Creates a Bottleneck with conv 1x1->3x3->1x1 layers.

        Note:
          1. Addition of feature maps occur at just before the final ReLU with the input feature maps
          2. if input size is different from output, select projected mapping or else identity mapping.
          3. if is_Bottleneck=False (3x3->3x3) are used else (1x1->3x3->1x1). Bottleneck is required for resnet-50/101/152
        Args:
            in_channels (int) : input channels to the Bottleneck
            intermediate_channels (int) : number of channels to 3x3 conv
            expansion (int) : factor by which the input #channels are increased
            stride (int) : stride applied in the 3x3 conv. 2 for first Bottleneck of the block and 1 for remaining

        Attributes:
            Layer consisting of conv->batchnorm->relu

        """

        super(ResidualBlock, self).__init__()

        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_bottle_neck = is_bottle_neck

        if self.in_channels == self.intermediate_channels*self.expansion:
            self.identity = True
        else:
            self.identity = False
            projection_layer = []
            projection_layer.append(nn.Conv2d(in_channels = self.in_channels, out_channels=self.intermediate_channels*self.expansion, kernel_size=1, stride=stride, padding=0, bias=False))
            projection_layer.append(nn.BatchNorm2d(self.intermediate_channels*self.expansion))
            self.projection = nn.Sequential(*projection_layer)

        self.relu = nn.ReLU()

        if self.is_bottle_neck:
            self.conv1_1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.batchNorm_1 = nn.BatchNorm2d(self.intermediate_channels)

            self.conv2_3x3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.batchNorm_2 = nn.BatchNorm2d(self.intermediate_channels)

            self.conv3_1x1 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=intermediate_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
            self.batchNorm_3 = nn.BatchNorm2d(self.intermediate_channels*self.expansion)
        else:
            self.conv1_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.batchNorm_1 = nn.BatchNorm2d(self.intermediate_channels)

            self.conv2_3x3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchNorm_2 = nn.BatchNorm2d(self.intermediate_channels)

    def forward(self, x):
        in_x = x

        if self.is_bottle_neck:
            x = self.relu(self.batchNorm_1(self.conv1_1x1(x)))

            x = self.relu(self.batchNorm_2(self.conv2_3x3(x)))

            x = self.batchNorm_3(self.conv3_1x1(x))

        else:
            x = self.relu(self.batchNorm_1(self.conv1_3x3(x)))

            x = self.batchNorm_2(self.conv2_3x3(x))

        if(self.identity):
            x += in_x
        else:
            x += self.projection(in_x)

        x = self.relu(x)

        return x


def test_ResidualBlock():
    x = torch.randn(1,64,112,112)
    model = ResidualBlock(64,64,4,True,2)
    print(model(x).shape)
    del model

test_ResidualBlock()