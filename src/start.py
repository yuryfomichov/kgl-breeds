import torch as torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.trainer.trainer import Trainer
from torch.utils.trainer.plugins import *
from src.dataloader import BreedsLoader
from src.model import BreedsModel
from src.validationplugin import ValidationPlugin
from src.saverplugin import SaverPlugin
import os
import re
import glob
from natsort import natsorted
import pandas as pd

data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def main():
    loss_fn = nn.CrossEntropyLoss().type(data_type)
    loader = BreedsLoader({
        'batch_size': 250
    })
    model = BreedsModel()
    model = model.type(data_type)
    optimizer = optim.Adam(model.parameters())
    trainer = Trainer(model=model, dataset=loader.get_train_loader(), criterion=loss_fn, optimizer=optimizer)
    trainer.cuda = True if torch.cuda.is_available() else False
    trainer.register_plugin(AccuracyMonitor())
    trainer.register_plugin(LossMonitor())
    trainer.register_plugin(ProgressMonitor())
    trainer.register_plugin(TimeMonitor())
    trainer.register_plugin(ValidationPlugin(loader.get_val_loader(), loader.get_val_loader()))
    trainer.register_plugin(SaverPlugin('checkpoints/', False))
    trainer.register_plugin(Logger(['accuracy', 'loss', 'progress', 'time','validation_loss', 'test_loss']))
    trainer.run(epochs=5)
    checkpoint_data = load_last_checkpoint('checkpoints')
    # if checkpoint_data is not None:
    #     (state_dict, epoch, iteration) = checkpoint_data
    #     trainer.epochs = epoch
    #     trainer.iterations = iteration
    #     trainer.model.load_state_dict(state_dict)


def load_last_checkpoint(checkpoints_path):
    checkpoints_pattern = os.path.join(
        checkpoints_path, SaverPlugin.last_pattern.format('*', '*')
    )
    checkpoint_paths = natsorted(glob.glob(checkpoints_pattern))
    if len(checkpoint_paths) > 0:
        checkpoint_path = checkpoint_paths[-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(SaverPlugin.last_pattern.format(r'(\d+)', r'(\d+)'), checkpoint_name)
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return (torch.load(checkpoint_path), epoch, iteration)
    else:
        return None

def get_submission():
    (state_dict, epoch, iteration) = load_last_checkpoint('checkpoints')
    model = BreedsModel()
    model.load_state_dict(state_dict)
    loader = BreedsLoader({'batch_size': 250, 'shuffle': False})
    df = pd.DataFrame(columns=loader.get_breeds())
    model.eval()
    for x, y in loader.get_submission_loader():
        x_var = Variable(x.type(data_type), volatile=True)
        scores = model(x_var)
        probabitilty = nn.Softmax()(scores).data.cpu().numpy()
        df = df.append(pd.DataFrame(probabitilty, columns=loader.get_breeds()))
    df['id'] = np.char.replace(os.listdir('data/test'), '.jpg', '')
    df = df.reindex_axis(['id'] + loader.get_breeds(), axis=1)
    df.to_csv('submission.csv', index = False)

main()
get_submission()
