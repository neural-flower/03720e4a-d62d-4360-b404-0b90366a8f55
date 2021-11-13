import os.path
from PIL import Image
import src.drone_cnn as dc
from src.SunFlowerDataset import SunFlowerDataset


def main():
    # Training
    cnn = dc.DroneTrainer('annotations.xml', '~/Downloads/fuckyou/images')
    cnn.train(10)

    # Testing
    # cnn = dc.DroneCnn(2, 'resources/parameters-4')
    # dst = DroneDataset('resources/train', dc.get_transform(train=False))
    # cnn.process_image(dst, 0)


if __name__ == "__main__":
    main()