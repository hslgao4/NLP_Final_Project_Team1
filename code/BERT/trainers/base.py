from .loggers import *
from .utils import AverageMeterSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path
from abc import ABC, abstractmethod

class AbstractTrainer(ABC):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.log_period_as_iter = args.log_period_as_iter
    
    def _create_loggers(self):
            """Creates logging utilities for training."""
            root = Path(self.export_root)
            writer = SummaryWriter(log_dir=root.joinpath('logs'))

            # Creating model checkpoints directories
            model_checkpoint_path = root.joinpath('models')
            model_checkpoint_path.mkdir(exist_ok=True)

            # Setting up train loggers
            train_loggers = [
                MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train')
            ]

            # Setting up validation loggers
            val_loggers = [
                MetricGraphPrinter(writer, key='accuracy', graph_name='Accuracy', group_name='Validation')
            ]

            return writer, train_loggers, val_loggers

    def _create_optimizer(self):
        """Create and return an optimizer."""
        if self.args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        else:
            raise ValueError("Unsupported optimizer type provided!")

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass
    
    def validate(self, epoch, accum_iter):
        """Evaluate model performance on the validation dataset."""
        self.model.eval()  # Set model to evaluation mode
        average_meter_set = AverageMeterSet()
        with torch.no_grad():  # Disable gradient computation
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.calculate_loss(outputs, targets)
                average_meter_set.update('validation_loss', loss.item(), n=inputs.size(0))

                # Calculate additional metrics if needed
                metrics = self.calculate_metrics(outputs, targets)
                for metric_name, metric_value in metrics.items():
                    average_meter_set.update(metric_name, metric_value)

        # Logging and printing validation results
        self.log_extra_val_info({
            'epoch': epoch,
            'accum_iter': accum_iter,
            'loss': average_meter_set.averages()['validation_loss'],
            **{k: v for k, v in average_meter_set.averages().items() if k != 'validation_loss'}
        })

        # Reset model to training mode
        self.model.train()
    def train(self, epoch=None):  # Now accepts an epoch number, but it's optional
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                f'Epoch {epoch+1}, loss {average_meter_set["loss"].avg:.3f}')

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter
