import glob
import os
import re
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
import torch.optim as optim
from natsort import natsorted
from torch.autograd import Variable
from dataloader import BreedsLoader
from model import BreedsModel
from trainer.trainer import BreedsTrainer
from trainer.plugins.saverplugin import SaverPlugin

data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def main():
    print('IsCuda', torch.cuda.is_available())
    loss_fn = nn.CrossEntropyLoss().type(data_type)
    loader = BreedsLoader({
        'batch_size': 150
    })
    model = BreedsModel().type(data_type)
    optimizer = optim.Adam(model.parameters())
    trainer = BreedsTrainer(model, loader, loss_fn, optimizer)
    trainer.run(lrs=[1e-3, 5e-4, 2e-4, 1e-4], epochs=[1,1,1,1])
    #checkpoint_data = load_last_checkpoint('checkpoints')
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
    model = model.type(data_type)
    model.load_state_dict(state_dict)
    loader = BreedsLoader({'batch_size': 150, 'shuffle': False})
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
