from .basic_conv_autoencoder import ConvAutoencoder
from .resnet_autoencoder import *

def build_autoencoder(args):
    if args.model == "basic_conv":
        return ConvAutoencoder()
    elif "resnet" in args.model:
        configs, bottleneck = get_configs(args.model)
        return ResNetAutoEncoder(configs=configs, bottleneck=bottleneck)
    
