from .basic_conv_autoencoder import ConvAutoencoder
from .basic_conv_autoencoder import get_configs as get_basic_conv_configs
from .resnet_autoencoder import *

def build_autoencoder(args):
    if "basic_conv" in args.model:
        configs = get_basic_conv_configs(args.model)
        return ConvAutoencoder(configs=configs)
    elif "resnet" in args.model:
        configs, bottleneck = get_configs(args.model)
        return ResNetAutoEncoder(configs=configs, bottleneck=bottleneck, pretrained=True)
    
