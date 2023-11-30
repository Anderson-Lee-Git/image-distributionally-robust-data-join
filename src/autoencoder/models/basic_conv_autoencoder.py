# https://github.com/oke-aditya/image_similarity/blob/master/image_similarity/torch_model.py
import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    """
    A simple Convolutional Encoder Model
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        # self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        # self.relu5 = nn.ReLU(inplace=True)
        # self.maxpool5 = nn.MaxPool2d((2, 2))

        self._initialize_weights()

    def forward(self, x):
        # Downscale the image with conv maxpool etc.
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # print(x.shape)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # print(x.shape)

        # x = self.conv5(x)
        # x = self.relu5(x)
        # x = self.maxpool5(x)

        # print(f"Encoder output shape: {x.shape}")
        return x
    
    def _initialize_weights(self):
        for param in self.parameters():
            torch.nn.init.kaiming_normal_(param)


class ConvDecoder(nn.Module):
    """
    A simple Convolutional Decoder Model
    """

    def __init__(self):
        super().__init__()
        # self.deconv1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        # # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        # self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))
        self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu5 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        # print(x.shape)
        # x = self.deconv1(x)
        # x = self.relu1(x)
        # print(x.shape)

        x = self.deconv2(x)
        x = self.relu2(x)
        # print(x.shape)

        x = self.deconv3(x)
        x = self.relu3(x)
        # print(x.shape)

        x = self.deconv4(x)
        x = self.relu4(x)
        # print(x.shape)

        x = self.deconv5(x)
        x = self.relu5(x)
        # print(x.shape)
        return x
    
    def _initialize_weights(self):
        for param in self.parameters():
            torch.nn.init.kaiming_normal_(param)