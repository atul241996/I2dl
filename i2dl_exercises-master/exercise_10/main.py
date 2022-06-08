import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning import Callback

from exercise_code.util import visualizer, save_model
from exercise_code.networks.segmentation_nn import SegmentationNN
from exercise_code.data.segmentation_dataset import SegmentationData
from exercise_code.data import segmentation_transforms


def evaluate_model(model, dataloader):
    test_scores = []
    model.eval()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(model.device), targets.to(model.device)

        outputs = model.forward(inputs)
        _, preds = torch.max(outputs, 1)
        targets_mask = targets >= 0
        test_scores.append(np.mean((preds.cpu() == targets.cpu())[targets_mask].numpy()))

    return np.mean(test_scores)


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


data_root = os.path.join('..', 'datasets', 'segmentation')
train_dataset = SegmentationData(os.path.join(data_root, 'segmentation_data', 'train.txt'))
val_dataset = SegmentationData(os.path.join(data_root, 'segmentation_data', 'val.txt'))
test_dataset = SegmentationData(os.path.join(data_root, 'segmentation_data', 'test.txt'))
fixed_hparams = {
    'data_root_path': data_root,
    'image_dims': (240, 240),
    'augment_train_data': True,
    'in_channels': 3,
    'channel_size': 32,
    'depth_multiplier': 3,
    'depth': 4,
    'dropout_rate': 0.0,
    'optim_interval': 4,
    'lr': 1e-4,
    'lr_decay_rate': 0.9,
    'batch_size': 11,
}


def load_model(model_path, fixed_hparams):
    with open(model_path, 'rb') as model_file:
        model_dict = pickle.load(model_file)
        state_dict, hparams = model_dict['state_dict'], model_dict['hparams']
        hparams.update(fixed_hparams)
        model = SegmentationNN(hparams=hparams)
        model.load_state_dict(state_dict)
        return model


def show_few_images(model, mode='train', how_many=3):
    to_pil = transforms.ToPILImage()
    fig, axs = plt.subplots(how_many, 2)
    fig.set_dpi(300)
    for i in range(how_many):
        image, segmentation = model.dataset[mode][i]
        image = to_pil(image)
        axs[i, 0].imshow(image)
        axs[i, 1].imshow(segmentation)
    fig.show()


def objective(trial):
    # as explained above, we'll use this callback to collect the validation accuracies
    metrics_callback = MetricsCallback()

    # create a trainer
    trainer = pl.Trainer(
        # train_percent_check=1.0,
        # val_percent_check=1.0,
        logger=True,  # deactivate PL logging
        max_epochs=50,  # epochs
        gpus=1 if torch.cuda.is_available() else None,  # #gpus
        callbacks=[metrics_callback],  # save latest accuracy
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='val_acc'),  # early stopping
    )

    # here we sample the hyper params, similar as in our old random search
    trial_hparams = {
        'channel_size': trial.suggest_int('channel_size', 8, 64, 4),
        'depth_multiplier': trial.suggest_int("depth_multiplier", 1, 4),
        'dropout_rate': trial.suggest_loguniform('dropout_rate', 1e-6, 5e-1),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'lr_decay_rate': trial.suggest_loguniform('lr_decay_rate', 1e-1, 1e0),
    }
    trial_hparams.update(fixed_hparams)

    # create model from these hyper params and train it
    model = SegmentationNN(hparams=trial_hparams)
    model.prepare_data()
    trainer.fit(model)

    # save model
    save_model(model, '{}.p'.format(trial.number), 'checkpoints')

    # return validation accuracy from latest model, as that's what we want to minimize by our hyper param search
    return metrics_callback.metrics[-1]['val_acc']


if __name__ == '__main__':
    if False:
        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective, n_trials=100, timeout=36000)

        print('Number of finished trials: {}'.format(len(study.trials)))

        print('Best trial:')
        best_trial = study.best_trial

        print('  Value: {}'.format(best_trial.value))

        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        model = load_model(os.path.join('checkpoints', '{}.p'.format(best_trial.number)), fixed_hparams)

    if True:
        # create a trainer
        trainer = pl.Trainer(
            # train_percent_check=1.0,
            # val_percent_check=1.0,
            logger=True,  # deactivate PL logging
            max_epochs=100,  # epochs
            gpus=1 if torch.cuda.is_available() else None,  # #gpus
        )
        # here we sample the hyper params, similar as in our old random search
        trial_hparams = {}
        trial_hparams.update(fixed_hparams)

        # create model from these hyper params and train it
        model = SegmentationNN(hparams=trial_hparams)
        model_path = os.path.join('checkpoints', '{}.p'.format(0))
        if os.path.exists(model_path):
            model = load_model(model_path, fixed_hparams)
        model.prepare_data()
        # show_few_images(model)
        trainer.fit(model)
        # save model
        save_model(model, '{}.p'.format(0), 'checkpoints')
        pass
