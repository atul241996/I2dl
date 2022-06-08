import matplotlib.pyplot as plt
import numpy as np
import os

from exercise_code.networks.layer import (
    Sigmoid,
    Relu,
    LeakyRelu,
    Tanh,
)
from exercise_code.data import (
    DataLoader,
    ImageFolderDataset,
    RescaleTransform,
    NormalizeTransform,
    FlattenTransform,
    ComposeTransform,
)
from exercise_code.networks import (
    ClassificationNet,
    BCE,
    CrossEntropyFromLogits
)
from exercise_code.solver import Solver

import math
import hyperopt
from hyperopt import hp


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

download_url = "https://cdn3.vision.in.tum.de/~dl4cv/cifar10.zip"
i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
cifar_root = os.path.join(i2dl_exercises_path, "datasets", "cifar10")
# Use the Cifar10 mean and standard deviation computed in Exercise 3.
cifar_mean = np.array([0.49191375, 0.48235852, 0.44673872])
cifar_std  = np.array([0.24706447, 0.24346213, 0.26147554])
# Define all the transforms we will apply on the images when
# retrieving them.
rescale_transform = RescaleTransform()
normalize_transform = NormalizeTransform(
    mean=cifar_mean,
    std=cifar_std
)
flatten_transform = FlattenTransform()
compose_transform = ComposeTransform([rescale_transform,
                                      normalize_transform,
                                      flatten_transform])

# Create a train, validation and test dataset.
datasets = {}
for mode in ['train', 'val', 'test']:
    crt_dataset = ImageFolderDataset(
        mode=mode,
        root=cifar_root,
        download_url=download_url,
        transform=compose_transform,
        split={'train': 0.6, 'val': 0.2, 'test': 0.2}
    )
    datasets[mode] = crt_dataset
dataloaders = {}
for mode in ['train', 'val', 'test']:
    crt_dataloader = DataLoader(
        dataset=datasets[mode],
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )
    dataloaders[mode] = crt_dataloader

overfit_dataset = ImageFolderDataset(
    mode='train',
    root=cifar_root,
    download_url=download_url,
    transform=compose_transform,
    limit_files=1000
)
dataloaders['train_small'] = DataLoader(
    dataset=overfit_dataset,
    batch_size=5,
    shuffle=True,
    drop_last=False,
)


class Evaluator:
    def __init__(self, train_loader, val_loader, epochs, patience, model_class, integer_param_names=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.model_class = model_class

        if integer_param_names is None:
            integer_param_names = []
        self.integer_param_names = set(integer_param_names)

    def __call__(self, config):
        for p in self.integer_param_names:
            config[p] = int(config[p])

        print("\nEvaluating Config:\n", config)

        if 'activation' in config:
            if config['activation'] == 0:
                config['activation'] = Relu()
            elif config['activation'] == 1:
                config['activation'] = LeakyRelu(0.01)
            else:
                raise NotImplementedError('Activation function not available: %s' % str(config['activation']))
        config['loss_func'] = CrossEntropyFromLogits()

        model = self.model_class(**config)
        solver = Solver(model, self.train_loader, self.val_loader, **config)
        solver.train(epochs=self.epochs, patience=self.patience)

        return {'status': hyperopt.STATUS_OK, 'loss': solver.best_model_stats["val_loss"], 'config': config}


space = {
    'num_layer': hp.quniform('num_layer', 2.5, 5.5, 1),                               # (3, 5)
    'hidden_size': hp.quniform('hidden_size', 45, 255, 10),                           # (50, 250, 10)
    'learning_rate': hp.loguniform('learning_rate', math.log(1e-5), math.log(1e-2)),  # (1e-5, 1e-2)
    'lr_decay': hp.uniform('lr_decay', 0.5, 1),                                       # (0.5, 1)
    'reg': hp.loguniform('reg', math.log(1e-5), math.log(5e-1)),                      # (1e-5, 5e-1)
    'activation': hp.choice('activation', [0, 1]),
}

trials = hyperopt.Trials()
# Change here if you want to use the full training set
use_full_training_set = True
if not use_full_training_set:
    train_loader = dataloaders['train_small']
else:
    train_loader = dataloaders['train']
best_params = hyperopt.fmin(
    Evaluator(train_loader, dataloaders['val'], epochs=30, patience=5,
              model_class=ClassificationNet, integer_param_names=['num_layer', 'hidden_size']),
    space, algo=hyperopt.atpe.suggest, max_evals=50, trials=trials)

pass
