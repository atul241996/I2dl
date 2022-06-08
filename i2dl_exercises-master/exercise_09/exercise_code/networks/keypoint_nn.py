"""Models for facial keypoint detection"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from PIL import Image

from exercise_code.data.facial_keypoints_dataset import FacialKeypointsDataset
from exercise_code.data import facial_transforms


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        def size_after_conv2d(h_in, kernel_size=3, stride=1, padding=1, dilation=1):
            return math.floor((h_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
            elif type(m) == nn.Conv2d:
                torch.nn.init.xavier_normal_(m.weight)
            elif type(m) == nn.BatchNorm1d or type(m) == nn.BatchNorm2d:
                m.bias.data.fill_(0.001)

        self.augment_train_data = self.hparams.get('augment_train_data', True)
        image_dims = self.hparams['image_dims']
        in_channels = self.hparams['in_channels']
        flat_output_size = self.hparams['flat_output_size']
        # range is [16, 256]
        channel_size = self.hparams.get('channel_size', 24)
        # activation can be either 'LeakyReLU', 'ReLU' or 'PReLU'
        activation = self.hparams.get('activation', 'PReLU')
        # layer_multiplier has to be at least 1. range is [1,3]
        layer_multiplier = self.hparams.get('layer_multiplier', 2)
        # halving_order is the number of times image is halved. range is [0,5]
        halving_order = self.hparams.get('halving_order', 4)
        # range is [32, 1024]
        ff_hidden_size = self.hparams.get('ff_hidden_size', 1024)
        # range is [0, 0.5]
        dropout_rate = self.hparams.get('dropout_rate', 0)
        if activation == 'LeakyReLU':
            act_func = nn.LeakyReLU()
        elif activation == 'ReLU':
            act_func = nn.ReLU()
        else:
            act_func = nn.PReLU()

        # range is (0, 1]
        self.lr = hparams.get('lr', 1e-3)
        # range is (0, 1]
        self.lr_decay_rate = hparams.get('lr_decay_rate', 0.5)
        # range is [32, 512]
        self.batch_size = self.hparams.get('batch_size', 128)

        layers = []
        for before_halving_order in range(layer_multiplier):
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                # nn.BatchNorm2d(num_features=channel_size),
                act_func,
                nn.Dropout2d(p=dropout_rate)])
            image_dims = tuple(size_after_conv2d(h_in, kernel_size=3, stride=1, padding=1, dilation=1) for h_in in image_dims)
            in_channels = channel_size
        for layer_order in range(halving_order):
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=(layer_order + 2) * channel_size, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
                # nn.BatchNorm2d(num_features=(layer_order + 2) * channel_size),
                act_func,
                nn.Dropout2d(p=dropout_rate)])
            image_dims = tuple(size_after_conv2d(h_in, kernel_size=3, stride=2, padding=1, dilation=1) for h_in in image_dims)
            in_channels = (layer_order + 2) * channel_size
            for _ in range(layer_multiplier - 1):
                layers.extend([
                    nn.Conv2d(in_channels=in_channels, out_channels=(layer_order + 2) * channel_size, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                    # nn.BatchNorm2d(num_features=(layer_order + 2) * channel_size),
                    act_func,
                    nn.Dropout2d(p=dropout_rate)])
                image_dims = tuple(size_after_conv2d(h_in, kernel_size=3, stride=1, padding=1, dilation=1) for h_in in image_dims)
        layers.extend([
            nn.Flatten(),
            nn.Linear(in_features=(halving_order + 1) * channel_size * image_dims[0] * image_dims[1], out_features=ff_hidden_size, bias=True),
            # nn.BatchNorm1d(num_features=ff_hidden_size),
            act_func,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=ff_hidden_size, out_features=flat_output_size, bias=True),
            nn.Tanh()])
        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        # load to device!
        x = x.to(self.device)

        # feed x into model!
        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx):
        images, keypoints = batch['image'], batch['keypoints']

        # reshape keypoints from (N,15,2) to (N,30)
        keypoints = keypoints.reshape(keypoints.shape[0], -1)

        # load X, y to device!
        images, keypoints = images.to(self.device), keypoints.to(self.device)

        # forward pass
        out = self.forward(images)

        # loss
        loss = nn.functional.mse_loss(out, keypoints)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        tensorboard_logs = {'training_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {'test_loss': loss}

    def general_end(self, outputs, mode):
        mode_loss = mode + '_loss'
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode_loss] for x in outputs]).mean()
        return avg_loss

    def validation_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")
        val_score = 1 / (2 * avg_loss)
        tensorboard_logs = {'val_score': val_score}
        return {'val_loss': avg_loss, 'val_score': val_score, 'log': tensorboard_logs}

    def prepare_data(self):
        # create dataset
        FACIAL_ROOT = self.hparams['data_root_path']

        train_augment_transforms = []
        if self.augment_train_data:
            train_augment_transforms = [
                facial_transforms.FacialRandomHorizontalFlip(p=0.5),
                facial_transforms.FacialColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                facial_transforms.FacialRandomAffine(degrees=(-15, 15), translate=(0.06, 0.03), resample=Image.BILINEAR)
            ]
        # mean = self.hparams['in_channels'] * [0.5]
        # std = self.hparams['in_channels'] * [0.25]
        training_transforms = facial_transforms.FacialCompose(
            train_augment_transforms +
            [
                facial_transforms.FacialToTensor(),
                # facial_transforms.FacialNormalize(mean, std, inplace=False)
            ])
        val_transforms = facial_transforms.FacialCompose([
            facial_transforms.FacialToTensor(),
            # facial_transforms.FacialNormalize(mean, std, inplace=False)
        ])

        train_dataset = FacialKeypointsDataset(
            train=True,
            transform=training_transforms,
            root=FACIAL_ROOT,
        )
        val_dataset = FacialKeypointsDataset(
            train=False,
            transform=val_transforms,
            root=FACIAL_ROOT,
        )

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"] = train_dataset, val_dataset

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.batch_size, num_workers=1)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.batch_size, num_workers=1)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_decay_rate, patience=1)
        config = ([optimizer],
                  [{
                      'scheduler': lr_scheduler,  # The LR schduler
                      'interval': 'epoch',  # The unit of the scheduler's step size
                      'frequency': 1,  # The frequency of the scheduler
                      'reduce_on_plateau': True,  # For ReduceLROnPlateau scheduler
                      'monitor': 'val_loss'  # Metric to monitor
                  }])
        return config


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
