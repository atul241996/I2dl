import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np

class MyPytorchModel(pl.LightningModule):
    
    def __init__(self, hparams, input_size=3 * 32 * 32, num_classes=10):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.model = None 

        ########################################################################
        # TODO: Initialize your model!                                         #
        ########################################################################
        min_n_layers = 1

        self.hidden_size = hparams.get('hidden_size', 128)
        self.num_classes = num_classes
        self.n_layers = max(hparams.get('n_layers', 3), min_n_layers)
        self.dropout_rate = hparams.get('dropout_rate', 0.1)
        self.activation = hparams.get('activation', 'PReLU')
        if self.activation == 'LeakyReLU':
            act_func = nn.LeakyReLU()
        elif self.activation == 'ReLU':
            act_func = nn.ReLU()
        else:
            act_func = nn.PReLU()

        self.lr = hparams.get('lr', 1e-1)
        self.lr_decay_rate = hparams.get('lr_decay_rate', 0.1)

        layers = [nn.Linear(input_size, self.hidden_size),
                  nn.Dropout(p=self.dropout_rate),
                  act_func,
                  nn.BatchNorm1d(self.hidden_size)]
        for _ in range(self.n_layers - 1):
            layers.extend([nn.Linear(self.hidden_size, self.hidden_size),
                           nn.Dropout(p=self.dropout_rate),
                           act_func,
                           nn.BatchNorm1d(self.hidden_size)])
        layers.append(nn.Linear(self.hidden_size, self.num_classes))
        self.model = nn.Sequential(*layers)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):

        # x.shape = [batch_size, 3, 32, 32] -> flatten the image first
        x = x.view(x.shape[0], -1)
        
        # load to device!
        x = x.to(self.device)

        # feed x into model!
        x = self.model(x)

        return x
    
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # load X, y to device!
        images, targets = images.to(self.device), targets.to(self.device)

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct':n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct':n_correct}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        #print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def prepare_data(self):

        # create dataset
        CIFAR_ROOT = "../datasets/cifar10"
        my_transform = None
        ########################################################################
        # TODO: Define your transforms (convert to tensors, normalize).        #
        # If you want, you can also perform data augmentation!                 #
        ########################################################################

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        my_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=False)
        ])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        cifar_complete = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=my_transform)
        torch.manual_seed(0)
        N = len(cifar_complete)
        cifar_train, cifar_val, cifar_test = torch.utils.data.random_split(cifar_complete, 
                                                                           [int(N*0.6), int(N*0.2), int(N*0.2)])
        torch.manual_seed(torch.initial_seed())

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = cifar_train, cifar_val, cifar_test

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.hparams["batch_size"], num_workers=1)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"], num_workers=1)
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"], num_workers=1)

    def configure_optimizers(self):

        optim = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optim = ([optimizer],
                 [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=self.lr_decay_rate, patience=1)])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return optim

    def getTestAcc(self, loader = None):
        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc