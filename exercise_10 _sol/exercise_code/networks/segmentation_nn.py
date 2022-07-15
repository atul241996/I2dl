"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models, transforms

class SegmentationNN(pl.LightningModule):
# class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

        # use the pretrained features
        features = models.mobilenet_v2(pretrained=True).features
        for param in features.parameters():
            param.requires_grad = False
        
        self.filters = self.hparams['filters']
        # self.batch_size = self.hparams['batch_size']

        self.model = nn.Sequential(
            *list(features.children()),
            # AlexNet -> [256,6,6]
            # MobileNet_v2 -> [1280,8,8]
            # out = (in + 2*pad - kernel)/stride + 1
            #   for kernel=3 && padding=1 && stride=1, out=in

            # 8 -> 15 -> 15
            nn.Upsample(scale_factor=15/8),
            nn.Conv2d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.filters[1]),
            nn.ReLU(),

            # 15 -> 30 -> 30
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.filters[2]),
            nn.ReLU(),

            # 30 -> 60 -> 60
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.filters[2], out_channels=self.filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.filters[3]),
            nn.ReLU(),

            # out = (in - 1)*stride - 2*pad + kernel + output_padding
            # 60 -> (60 - 1)*2 - 2 + 3 + 1 = 120
            nn.ConvTranspose2d(in_channels=self.filters[3], out_channels=self.filters[4], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.filters[4]),
            nn.ReLU(),

            # 120 -> 240
            nn.ConvTranspose2d(in_channels=self.filters[4], out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        self.model = self.model.to(device)
        x = self.model(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    def general_step(self, batch, batch_idx, mode):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images, targets = batch[0], batch[1]
        images, targets = images.to(device), targets.to(device)
        outputs = self.forward(images)
        loss = self.loss_func(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def configure_optimizers(self):
        LR = self.hparams['lr']
        optim = torch.optim.Adam(self.model.parameters(), LR)
        # optim = torch.optim.SGD(self.model.parameters(), LR, momentum=0.9, weight_decay=1e-4)
        return optim

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