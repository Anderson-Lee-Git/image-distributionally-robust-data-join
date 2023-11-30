import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class InverseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, upsample = None):
        super(ResidualBlock, self).__init__()
        self.dconv1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.dconv2 = nn.Sequential(
                        nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.upsample = upsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        if self.upsample:
            residual = self.upsample(x)
        residual = x
        out = self.dconv1(x)
        out = self.dconv2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, embed_dim = 512):
        super(ResNetEncoder, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        print("Encoder")
        x = self.conv1(x)  # 64 x 32 x 32
        print(f"conv1: {x.shape}")
        x = self.maxpool(x)  # 64 x 16 x 16
        print(f"maxpool2d: {x.shape}")
        x = self.layer0(x)  # 64 x 16 x 16
        print(f"layer0: {x.shape}")
        x = self.layer1(x)  # 128 x 4 x 4
        print(f"layer1: {x.shape}")
        x = self.layer2(x)
        print(f"layer2: {x.shape}")
        x = self.layer3(x)
        print(f"layer3: {x.shape}")
        x = self.avgpool(x)
        print(f"output(avgpool) shape: {x.shape}")
        return x

class ResNetDecoder(nn.Module):
    def __init__(self, block, layers, embed_dim = 512):
        super(ResNetEncoder, self).__init__()
        self.inplanes = embed_dim
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 512, layers[0], kernel_size = 2, stride = 1, padding = 0)
        self.layer1 = self._make_layer(block, 256, layers[1], kernel_size = 2, stride = 2, padding = 0)
        self.layer2 = self._make_layer(block, 128, layers[2], kernel_size = 3, stride = 1, padding = 1)
        self.layer3 = self._make_layer(block, 64, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dconv = nn.Sequential(
                        nn.ConvTranspose2d(64, 3, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(3),
                        nn.ReLU())
        
    def _make_layer(self, block, planes, blocks, kernel_size=1, stride=1, padding=1):
        upsample = None
        # if stride != 1 or self.inplanes != planes:
        #     upsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
        #         nn.BatchNorm2d(planes),
        #     )
        layers = []
        layers.append(block(self.inplanes, planes, 
                            kernel_size=kernel_size,
                            stride=stride, padding=padding,
                            upsample=upsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        print("Decoder")
        x = self.layer0(x)
        print(f"layer0: {x.shape}")
        x = self.layer1(x)
        print(f"layer1: {x.shape}")
        x = self.layer2(x)
        print(f"layer2: {x.shape}")
        x = self.layer3(x)
        print(f"layer3: {x.shape}")
        x = self.dconv(x)
        print(f"output(dconv) shape: {x.shape}")
        return x