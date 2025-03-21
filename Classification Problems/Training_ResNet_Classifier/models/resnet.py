import torch
import torch.nn as nn
from torchsummary import summary

from residualBlock import ResidualBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# resnetX = (Num of channels, repetition, Bottleneck_expansion , Bottleneck_layer)
model_parameters={}
model_parameters['resnet18'] = ([64,128,256,512],[2,2,2,2],1,False)
model_parameters['resnet34'] = ([64,128,256,512],[3,4,6,3],1,False)
model_parameters['resnet50'] = ([64,128,256,512],[3,4,6,3],4,True)
model_parameters['resnet101'] = ([64,128,256,512],[3,4,23,3],4,True)
model_parameters['resnet152'] = ([64,128,256,512],[3,8,36,3],4,True)

class ResNet(nn.Module):
    def __init__(self, resnet_variant, in_channels, num_classes):
        """
        Creates the ResNet architecture based on the provided variant. 18/34/50/101 etc.
        Based on the input parameters, define the channels list, repeatition list along with expansion factor(4) and stride(3/1)
        using _make_blocks method, create a sequence of multiple Bottlenecks
        Average Pool at the end before the FC layer

        Args:
            resnet_variant (list) : eg. [[64,128,256,512],[3,4,6,3],4,True]
            in_channels (int) : image channels (3)
            num_classes (int) : output #classes

        Attributes:
            Layer consisting of conv->batchnorm->relu

        """
        super(ResNet, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repetition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_bottle_neck = resnet_variant[3]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_block(64, self.channels_list[0], self.repetition_list[0], self.expansion, self.is_bottle_neck, stride=1)
        self.block2 = self._make_block(self.channels_list[0]*self.expansion, self.channels_list[1], self.repetition_list[1], self.expansion, self.is_bottle_neck, stride=2)
        self.block3 = self._make_block(self.channels_list[1]*self.expansion, self.channels_list[2], self.repetition_list[2], self.expansion, self.is_bottle_neck, stride=2)
        self.block4 = self._make_block(self.channels_list[2]*self.expansion, self.channels_list[3], self.repetition_list[3], self.expansion, self.is_bottle_neck, stride=2)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.channels_list[3]*self.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x


    def _make_block(self, in_channels, intermediate_channels, num_repeat, expansion, is_bottle_neck, stride):
        """
        Args:
            in_channels : #channels of the Bottleneck input
            intermediate_channels : #channels of the 3x3 in the Bottleneck
            num_repeat : #Bottlenecks in the block
            expansion : factor by which intermediate_channels are multiplied to create the output channels
            is_Bottleneck : status if Bottleneck in required
            stride : stride to be used in the first Bottleneck conv 3x3

        Attributes:
            Sequence of Bottleneck layers

        """
        layers = []

        layers.append(ResidualBlock(in_channels, intermediate_channels, expansion, is_bottle_neck, stride=stride))

        for num in range(1, num_repeat):
            layers.append(ResidualBlock(intermediate_channels*expansion, intermediate_channels, expansion, is_bottle_neck,stride=1))

        return nn.Sequential(*layers)

# # Test model tự xây dựng
# architecture = 'resnet50'
# model = ResNet(model_parameters[architecture], in_channels=3, num_classes=1000).to(device)
# summary(model, (3, 224, 224), device="cuda")
#
# # Test model từ torchvision
# import torchvision
# from torchvision.models import resnet50, ResNet50_Weights
#
# torchvision_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
# summary(torchvision_model, (3, 224, 224), device="cuda")
