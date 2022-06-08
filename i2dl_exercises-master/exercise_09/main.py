import pickle
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning import Callback

from exercise_code.util.save_model import save_model
from exercise_code.networks.keypoint_nn import KeypointModel
from exercise_code.util import show_all_keypoints
from exercise_code.networks.keypoint_nn import DummyKeypointModel
from exercise_code.data.facial_keypoints_dataset import FacialKeypointsDataset
import exercise_code.data.facial_transforms as facial_transforms


def show_keypoint_predictions(model, dataset, num_samples=3):
    model.eval()
    for i in range(num_samples):
        image = dataset[i]["image"]
        key_pts = dataset[i]["keypoints"]
        with torch.no_grad():
            predicted_keypoints = torch.squeeze(model(image.unsqueeze(0))).view(15, 2)
        show_all_keypoints(image, key_pts, predicted_keypoints.to('cpu'))
    model.train()


def evaluate_model(model, dataset):
    model.freeze()
    criterion = F.mse_loss
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0
    for batch in dataloader:
        image, keypoints = batch["image"], batch["keypoints"]
        predicted_keypoints = model(image).view(-1, 15, 2)
        loss += criterion(
            torch.squeeze(keypoints),
            torch.squeeze(predicted_keypoints)
        ).item()
    model.unfreeze()
    return 1.0 / (2 * (loss/len(dataloader)))


def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model_dict = pickle.load(model_file)
        state_dict, hparams = model_dict['state_dict'], model_dict['hparams']
        hparams.update(fixed_hparams)
        model = KeypointModel(hparams)
        model.load_state_dict(state_dict)
        return model


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


data_root = '..\\datasets\\facial_keypoints'
# mean = 1 * [0.5]
# std = 1 * [0.25]
train_dataset = FacialKeypointsDataset(
    train=True,
    transform=facial_transforms.FacialCompose([
        facial_transforms.FacialToTensor(),
        # facial_transforms.FacialNormalize(mean, std, inplace=False)
    ]),
    root=data_root,
)
val_dataset = FacialKeypointsDataset(
    train=False,
    transform=facial_transforms.FacialCompose([
        facial_transforms.FacialToTensor(),
        # facial_transforms.FacialNormalize(mean, std, inplace=False)
    ]),
    root=data_root,
)
fixed_hparams = {
    'augment_train_data': True,
    'data_root_path': data_root,
    'image_dims': (96, 96),
    'in_channels': 1,
    'flat_output_size': 30,
    'batch_size': 343,
    'channel_size': 24,
    'activation': 'PReLU',
    'layer_multiplier': 2,
    'halving_order': 4,
    'ff_hidden_size': 1024,
    'dropout_rate': 0,
    'lr': 1e-5,
    'lr_decay_rate': 0.999999999,
}


def objective(trial):
    # as explained above, we'll use this callback to collect the validation accuracies
    metrics_callback = MetricsCallback()

    # create a trainer
    trainer = pl.Trainer(
        # train_percent_check=1.0,
        # val_percent_check=1.0,
        logger=True,  # deactivate PL logging
        max_epochs=200,  # epochs
        gpus=1 if torch.cuda.is_available() else None,  # #gpus
        callbacks=[metrics_callback],  # save latest accuracy
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='val_score'),  # early stopping
    )

    # here we sample the hyper params, similar as in our old random search
    trial_hparams = {
        # 'channel_size': trial.suggest_int('channel_size', 16, 64, 4),
        # 'activation': trial.suggest_categorical('activation', ('PReLU', 'ReLU', 'LeakyReLU')),
        # 'layer_multiplier': trial.suggest_int("layer_multiplier", 1, 2),
        # 'dropout_rate': trial.suggest_loguniform('dropout_rate', 1e-6, 5e-1),
        # 'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        # 'lr_decay_rate': trial.suggest_loguniform('lr_decay_rate', 1e-1, 1e0),
    }
    trial_hparams.update(fixed_hparams)

    # create model from these hyper params and train it
    model = KeypointModel(trial_hparams)
    model.prepare_data()
    trainer.fit(model)

    # save model
    save_model(model, '{}.p'.format(trial.number), 'checkpoints')

    # return validation accuracy from latest model, as that's what we want to minimize by our hyper param search
    return metrics_callback.metrics[-1]['val_score']


if __name__ == '__main__':
    if False:
        dummy = DummyKeypointModel()
        show_keypoint_predictions(dummy, train_dataset, 3)
        print("Score:", evaluate_model(dummy, val_dataset))

        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective, n_trials=1, timeout=36000)

        print('Number of finished trials: {}'.format(len(study.trials)))

        print('Best trial:')
        best_trial = study.best_trial

        print('  Value: {}'.format(best_trial.value))

        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        model = load_model(os.path.join('checkpoints', '{}.p'.format(best_trial.number)))

    # create a trainer
    trainer = pl.Trainer(
        # train_percent_check=1.0,
        # val_percent_check=1.0,
        logger=True,  # deactivate PL logging
        max_epochs=100,  # epochs
        gpus=1 if torch.cuda.is_available() else None,  # #gpus
    )
    # here we sample the hyper params, similar as in our old random search
    trial_hparams = {
        # 'channel_size': trial.suggest_int('channel_size', 16, 64, 4),
        # 'activation': trial.suggest_categorical('activation', ('PReLU', 'ReLU', 'LeakyReLU')),
        # 'layer_multiplier': trial.suggest_int("layer_multiplier", 1, 2),
        # 'dropout_rate': trial.suggest_loguniform('dropout_rate', 1e-6, 5e-1),
        # 'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        # 'lr_decay_rate': trial.suggest_loguniform('lr_decay_rate', 1e-1, 1e0),
    }
    trial_hparams.update(fixed_hparams)

    # create model from these hyper params and train it
    model = KeypointModel(trial_hparams)
    model_path = os.path.join('checkpoints', '{}.p'.format(0))
    if os.path.exists(model_path):
        model = load_model(model_path)
    model.prepare_data()
    trainer.fit(model)
    # save model
    save_model(model, '{}.p'.format(0), 'checkpoints')
    pass
