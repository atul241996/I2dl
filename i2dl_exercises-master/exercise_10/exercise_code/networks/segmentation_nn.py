"""SegmentationNN"""
import itertools
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from exercise_code.data.segmentation_dataset import SegmentationData
from exercise_code.data import segmentation_transforms


class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels, channel_size, depth, dropout_rate=0.0):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Conv2d(in_channels, channel_size, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_rate)])
            in_channels = channel_size
        self.model = nn.Sequential(*layers)
        self.downsampler = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        full_x = self.model(x)
        return self.downsampler(full_x), full_x


class ConvolutionalDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, channel_size, depth, dropout_rate=0.0, last_layer=False):
        super().__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.skip_channels = skip_channels
        in_channels = in_channels + skip_channels
        layers = []
        for _ in range(depth - 1):
            layers.extend([
                nn.Conv2d(in_channels, channel_size, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_rate)])
            in_channels = channel_size
        layers.append(nn.Conv2d(in_channels, channel_size, kernel_size=3, stride=1, padding=1, bias=last_layer))
        if not last_layer:
            layers.extend([
                nn.BatchNorm2d(channel_size),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_rate)])
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_x=None):
        x = self.upsampler(x)
        if self.skip_channels > 0:
            x = torch.cat([x, skip_x], dim=1)
        return self.model(x)


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_normal_(m.weight)
            elif type(m) == nn.BatchNorm2d:
                m.bias.data.fill_(0.001)

        # these have to be given!
        data_root_path = self.hparams['data_root_path']
        img_height, img_width = self.hparams['image_dims']
        self.augment_train_data = self.hparams.get('augment_train_data', True)
        in_channels = self.hparams['in_channels']
        # range is [16, 256]
        channel_size = self.hparams.get('channel_size', 16)
        # range is [1, 10]
        depth_multiplier = self.hparams.get('depth_multiplier', 2)
        # range is [1, 10]
        depth = self.hparams.get('depth', 4)
        # range is [0, 0.5]
        dropout_rate = self.hparams.get('dropout_rate', 0.0)

        # range is [1, 5]
        self.optim_interval = self.hparams.get('optim_interval', 1)
        # range is (0, 1]
        self.lr = hparams.get('lr', 5e-4) / self.optim_interval
        # range is (0, 1]
        self.lr_decay_rate = hparams.get('lr_decay_rate', 0.95)
        # range is [16, 512]
        self.batch_size = self.hparams.get('batch_size', 24)

        self.encoder_modules = nn.ModuleList()
        for i in range(1, depth + 1):
            self.encoder_modules.append(ConvolutionalEncoder(in_channels, i * channel_size, depth_multiplier, dropout_rate))
            in_channels = i * channel_size
        self.encoder_modules.apply(init_weights)
        self.decoder_modules = nn.ModuleList()
        for i in range(depth, 1, -1):
            self.decoder_modules.append(ConvolutionalDecoder(i * channel_size, i * channel_size, (i - 1) * channel_size, depth_multiplier, dropout_rate))
        self.decoder_modules.append(ConvolutionalDecoder(1 * channel_size, 1 * channel_size, num_classes, depth_multiplier, dropout_rate, True))
        self.decoder_modules.apply(init_weights)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        encoder_skips = []
        for encoder in self.encoder_modules:
            x, full_x = encoder(x)
            encoder_skips.append(full_x)

        for i, decoder in enumerate(self.decoder_modules, 1):
            x = decoder(x, encoder_skips[-i])

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    def general_step(self, batch, batch_idx):
        # target_labels.shape == (N, H, W)
        images, target_labels = batch

        # load X, y to device!
        images, target_labels = images.to(self.device), target_labels.to(self.device)

        # forward pass
        out = self.forward(images)

        # loss
        loss = self.loss_fn(out, target_labels)

        preds = out.argmax(axis=1)
        n_correct = (target_labels == preds).sum()
        return loss, n_correct

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx)
        tensorboard_logs = {'training_loss': loss}
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx)
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx)
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def general_end(self, outputs, mode):
        mode_loss, mode_n_correct = mode + '_loss', mode + '_n_correct'
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode_loss] for x in outputs]).mean()
        total_correct = torch.stack([x[mode_n_correct] for x in outputs]).sum().cpu().numpy()

        img_height, img_width = self.hparams['image_dims']
        acc = total_correct / (len(self.dataset[mode]) * img_height * img_width)
        return avg_loss, acc

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, 'val')
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if batch_idx % self.optim_interval == 0 or batch_idx == len(self.dataset['train']) - 1:
            optimizer.step()
            optimizer.zero_grad()

    def prepare_data(self):
        # create dataset
        data_root_path = self.hparams['data_root_path']

        train_augment_transforms = []
        if self.augment_train_data:
            train_augment_transforms = [
                segmentation_transforms.SegmentationRandomHorizontalFlip(p=0.5),
                segmentation_transforms.SegmentationColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                segmentation_transforms.SegmentationRandomNoise(noise_type='gauss'),
                segmentation_transforms.SegmentationRandomAffine(degrees=(-20, 20), translate=(0.07, 0.07))
            ]
        train_augment_transforms = segmentation_transforms.SegmentationCompose(train_augment_transforms)

        train_dataset = SegmentationData(os.path.join(data_root_path, 'segmentation_data', 'train.txt'), train_augment_transforms)
        val_dataset = SegmentationData(os.path.join(data_root_path, 'segmentation_data', 'val.txt'))
        test_dataset = SegmentationData(os.path.join(data_root_path, 'segmentation_data', 'test.txt'))

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = train_dataset, val_dataset, test_dataset

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.batch_size, num_workers=1)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.batch_size, num_workers=1)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size, num_workers=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            itertools.chain(self.encoder_modules.parameters(), self.decoder_modules.parameters()), lr=self.lr)
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

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
