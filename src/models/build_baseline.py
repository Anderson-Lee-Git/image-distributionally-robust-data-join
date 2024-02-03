from .resnet_autoencoder import get_configs,  ResNet

def build_resnet(num_classes, pretrained=False, args=None):
    if args is not None:
        configs, bottleneck = get_configs(args.model)
        model = ResNet(configs=configs, bottleneck=bottleneck, num_classes=num_classes,
                       pretrained=pretrained and args.model=="resnet50")
        return model
    else:
        raise NotImplementedError(f"Please provide model type in args")