# from decouple import config
from math import ceil
import torch
import torchvision
from PIL import Image
from torch import device
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import src.vision.transforms as T
import src.vision.utils as utils
from .SunFlowerDataset import SunFlowerDataset
from .vision import engine


def get_instance_segmentation_model(num_classes: int) -> MaskRCNN:
    # load an instance segmentation model pre-trained on COCO
    model: MaskRCNN = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train: bool) -> T.Compose:
    transforms = [T.ToTensor()]
    # converts the image, a PIL image, into a PyTorch Tensor
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class DroneTrainer:
    data_loader: DataLoader = None
    data_loader_t: DataLoader = None
    device: device = None
    model: MaskRCNN = None

    def __init__(self, csv_file: str, root_dir: str, num_classes: int = 2):
        """

        :param train_path: absolute path to folder with train data
        :param num_classes: dataset classes, 2 by default - background and object
        """

        self.csv_file = csv_file
        self.root_dir = root_dir
        # get the model using our helper function
        self.model = get_instance_segmentation_model(num_classes)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # apply model to device
        self.model.to(self.device)

        self.init_data_sets()

    def init_data_sets(self) -> None:
        # load train datasets
        ds = SunFlowerDataset(self.csv_file, self.root_dir, get_transform(True))
        ds_t = SunFlowerDataset(self.csv_file, self.root_dir, get_transform(False))

        torch.manual_seed(1)
        indices = torch.randperm(len(ds)).tolist()

        ds = Subset(ds, indices[:-ceil(len(ds) / 2)])
        ds_t = Subset(ds_t, indices[-ceil(len(ds) / 2):])

        self.data_loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
        self.data_loader_t = DataLoader(ds_t, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    def train(self, num_epochs: int) -> None:
        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            engine.train_one_epoch(self.model, optimizer, self.data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            engine.evaluate(self.model, self.data_loader_t, device=self.device)
            torch.save(self.model.state_dict(), 'resources/parameters-' + str(epoch))


class DroneCnn:

    def __init__(self, num_classes: int, state_dict_path):
        self.model = get_instance_segmentation_model(num_classes)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # apply model to device
        self.model.to(self.device)

        # load cnn parameters
        self.model.load_state_dict(torch.load(state_dict_path))

    # def process_image(self, dst: SunFlowerDataset, idx: int):
    #     image, _ = dst[idx]
    #     self.model.eval()
    #     with torch.no_grad():
    #         prediction = self.model([image.to(self.device)])
    #         print(prediction)
    #         srcImage = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
    #         OUTPUT = config('OUTPUT')
    #         srcImage.save(OUTPUT + 'src.jpg')
    #         image = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    #         image.save(OUTPUT + 'image.jpg')
