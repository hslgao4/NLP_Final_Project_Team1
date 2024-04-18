from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (seqs, labels) in enumerate(self.train_loader):
            seqs, labels = seqs.to(self.args.device), labels.to(self.args.device)
            self.optimizer.zero_grad()
            loss = self.calculate_loss((seqs, labels))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % self.log_period_as_iter == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

        # Additional logging or actions at the end of the epoch
        print(f'Epoch {epoch} completed, Average Loss: {total_loss / len(self.train_loader)}')

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        return self.ce(logits, labels)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics