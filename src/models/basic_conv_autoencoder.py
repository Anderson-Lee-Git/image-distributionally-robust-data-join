# https://github.com/oke-aditya/image_similarity/blob/master/image_similarity/torch_model.py
import torch
import torch.nn as nn
import math

from utils.init_utils import initialize_weights

def get_configs(arch="basic_conv_256"):
    configs = {}
    encoder = {}
    decoder = {}
    if arch == "basic_conv_256":
        encoder["channels"] = [[3, 16], [16, 32], [32, 64], [64, 128], [128, 256]]
        encoder["kernel_sizes"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        encoder["paddings"] = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        encoder["strides"] = [1, 1, 1, 1, 1]
        encoder["maxpools"] = [nn.Identity(),
                                nn.Identity(),
                                nn.MaxPool2d((2, 2)),
                                nn.Identity(),
                                nn.MaxPool2d((2, 2))]
        decoder["channels"] = [[256, 128], [128, 64], [64, 32], [32, 16], [16, 3]]
        decoder["kernel_sizes"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        decoder["paddings"] = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        decoder["strides"] = [1, 1, 1, 1, 1]
        decoder["upsamples"] = [nn.functional.interpolate,
                                nn.Identity(),
                                nn.functional.interpolate,
                                nn.Identity(),
                                nn.Identity()]
    elif arch == "basic_conv_512":
        encoder["channels"] = [[3, 16], [16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
        encoder["kernel_sizes"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        encoder["paddings"] = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        encoder["strides"] = [1, 1, 1, 1, 1, 1]
        encoder["maxpools"] = [nn.Identity(),
                                nn.Identity(),
                                nn.MaxPool2d((2, 2)),
                                nn.Identity(),
                                nn.MaxPool2d((2, 2)),
                                nn.Identity()]
        decoder["channels"] = [[512, 256], [256, 128], [128, 64], [64, 32], [32, 16], [16, 3]]
        decoder["kernel_sizes"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        decoder["paddings"] = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        decoder["strides"] = [1, 1, 1, 1, 1, 1]
        decoder["upsamples"] = [nn.Identity(),
                                nn.functional.interpolate,
                                nn.Identity(),
                                nn.functional.interpolate,
                                nn.Identity(),
                                nn.Identity()]
    elif arch == "basic_conv_256_4x4":
        encoder["channels"] = [[3, 16], [16, 32], [32, 64], [64, 128], [128, 256]]
        encoder["kernel_sizes"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        encoder["paddings"] = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        encoder["strides"] = [1, 1, 1, 2, 1]
        encoder["maxpools"] = [nn.Identity(),
                                nn.Identity(),
                                nn.MaxPool2d((2, 2)),
                                nn.Identity(),
                                nn.MaxPool2d((2, 2))]
        decoder["channels"] = [[256, 128], [128, 64], [64, 32], [32, 16], [16, 3]]
        decoder["kernel_sizes"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        decoder["paddings"] = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        decoder["strides"] = [1, 1, 1, 1, 1]
        decoder["upsamples"] = [nn.functional.interpolate,
                                nn.functional.interpolate,
                                nn.functional.interpolate,
                                nn.Identity(),
                                nn.Identity()]
    elif arch == "basic_conv_512_4x4":
        encoder["channels"] = [[3, 16], [16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
        encoder["kernel_sizes"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        encoder["paddings"] = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        encoder["strides"] = [1, 1, 1, 2, 1, 1]
        encoder["maxpools"] = [nn.Identity(),
                                nn.Identity(),
                                nn.MaxPool2d((2, 2)),
                                nn.Identity(),
                                nn.MaxPool2d((2, 2)),
                                nn.Identity()]
        decoder["channels"] = [[512, 256], [256, 128], [128, 64], [64, 32], [32, 16], [16, 3]]
        decoder["kernel_sizes"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        decoder["paddings"] = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        decoder["strides"] = [1, 1, 1, 1, 1, 1]
        decoder["upsamples"] = [nn.Identity(),
                                nn.functional.interpolate,
                                nn.functional.interpolate,
                                nn.functional.interpolate,
                                nn.Identity(),
                                nn.Identity()]
    configs["encoder"] = encoder
    configs["decoder"] = decoder
    return configs


class EncodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, activation=nn.Module, maxpool=nn.MaxPool2d((2, 2))):
        super().__init__()
        self.layer = nn.Conv2d(in_channel,
                               out_channel,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channel)
        self.activation = activation
        self.maxpool = maxpool
    
    def forward(self, x):
        # print(f"input shape: {x.shape}")
        x = self.layer(x)
        x = self.batch_norm(x)
        # print(f"after conv shape: {x.shape}")
        x = self.activation(x)
        x = self.maxpool(x)
        # print(f"after maxpool shape: {x.shape}")
        return x

class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, 
                 activation=nn.Module,
                 upsample=nn.functional.interpolate):
        super().__init__()
        self.layer = nn.ConvTranspose2d(in_channel,
                                        out_channel,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channel)
        self.activation = activation
        self.upsample = upsample
    
    def forward(self, x):
        # print(f"input shape: {x.shape}")
        x = self.layer(x)
        x = self.batch_norm(x)
        # print(f"after deconv shape: {x.shape}")
        x = self.activation(x)
        if isinstance(self.upsample, nn.Identity):
            x = self.upsample(x)
        else:
            x = self.upsample(x, scale_factor=2, mode="bilinear")
        # print(f"after upsample shape: {x.shape}")
        return x

class ConvEncoder(nn.Module):
    """
    A simple Convolutional Encoder Model
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.layers = self._build_layers()
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        # print(f"Encoder output shape: {x.shape}")
        return x

    def _build_layers(self):
        channels = self.configs["channels"]
        kernel_sizes = self.configs["kernel_sizes"]
        paddings = self.configs["paddings"]
        strides = self.configs["strides"]
        maxpools = self.configs["maxpools"]
        n_layers = len(channels)
        assert n_layers == len(channels) == len(kernel_sizes) == len(paddings) == len(strides)
        layers = []
        for i in range(n_layers):
            maxpool = maxpools[i]
            layers.append(EncodeBlock(
                channels[i][0],
                channels[i][1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                activation=nn.ReLU(),
                maxpool=maxpool
            ))
        layers = nn.Sequential(*layers)
        print(layers)
        return layers
    
    def _initialize_weights(self):
        initialize_weights({}, self.modules())

class ConvDecoder(nn.Module):
    """
    A simple Convolutional Decoder Model
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.layers = self._build_layers()
        self._initialize_weights()

    def forward(self, x):
        B, D = x.shape
        in_channel = self.configs["channels"][0][0]
        H = W = int(math.sqrt(D // in_channel))
        x = x.view(B, in_channel, H, W)
        x = self.layers(x)
        return x
    
    def _build_layers(self):
        channels = self.configs["channels"]
        kernel_sizes = self.configs["kernel_sizes"]
        paddings = self.configs["paddings"]
        strides = self.configs["strides"]
        upsamples = self.configs["upsamples"]
        n_layers = len(channels)
        assert n_layers == len(channels) == len(kernel_sizes) == len(paddings) == len(strides)
        layers = []
        for i in range(n_layers):
            layers.append(DecodeBlock(
                channels[i][0],
                channels[i][1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                activation=nn.ReLU(),
                upsample=upsamples[i]
            ))
        layers = nn.Sequential(*layers)
        print(layers)
        return layers

    def _initialize_weights(self):
        initialize_weights({}, self.modules())

class ConvAutoencoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.encoder = ConvEncoder(configs["encoder"])
        self.decoder = ConvDecoder(configs["decoder"])
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)
        return x