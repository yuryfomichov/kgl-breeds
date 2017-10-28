import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.trainer.trainer import Trainer
from torch.utils.trainer.plugins import AccuracyMonitor
from torch.utils.trainer.plugins import LossMonitor
from torch.utils.trainer.plugins import ProgressMonitor
from torch.utils.trainer.plugins import Logger
from torch.utils.trainer.plugins import TimeMonitor
from torch.optim import lr_scheduler
from src.dataloader import BreedsLoader
from src.model import BreedsModel

def main():
    data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    loss_fn = nn.CrossEntropyLoss().type(data_type)
    loader = BreedsLoader({
        'batch_size': 200
    })
    model = BreedsModel()
    optimizer = optim.Adam(model.parameters(), lr_scheduler)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    trainer = Trainer(model=model, dataset=loader.get_train_loader(), criterion=loss_fn, optimizer=optimizer)
    trainer.register_plugin(AccuracyMonitor())
    trainer.register_plugin(LossMonitor())
    trainer.register_plugin(ProgressMonitor())
    trainer.register_plugin(TimeMonitor())
    trainer.register_plugin(Logger(['accuracy', 'loss', 'progress', 'time']))
    trainer.run(epochs=3)

main()

