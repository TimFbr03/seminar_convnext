from train.convnet_training import train_cnn
from train.vit_training import train_vit
from train.convnext_training import train_convnext

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", help="Model type (eg. cnn, convnext, vit)")
    args = parser.parse_args()

    if args.model == "cnn":
        print("Traing ConvNet")
        train_cnn()

    if args.model == "convnext":
        print("Traing ConvNeXt")
        train_convnext()

    if args.model == "vit":
        print("Traing ViT")
        train_vit()
