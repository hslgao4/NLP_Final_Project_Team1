import os
import torch
from models.bert import BERTModel
from trainers.bert_trainer import BERTTrainer
from dataloader_BERT import UserInteractionDataset
from torch.utils.data import DataLoader


# Set environment variable for CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Config:
    bert_max_len = 512
    num_items = 10000  # adjust based on your dataset
    bert_num_blocks = 2
    bert_num_heads = 8
    bert_hidden_units = 768
    bert_dropout = 0.1
    model_init_seed = 42
    lr = 0.001
    num_epochs = 10
    train_batch_size = 32
    dataset_path = '/home/ubuntu/NLP_Project_Team1/data/user_sequences.parquet'
    mask_prob = 0.2
    mask_token = 999999
    num_workers = 4  # Number of workers for data loading
    export_root = '/home/ubuntu/NLP_Project_Team1/training_output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpu = torch.cuda.device_count()
    optimizer = 'adam'
    weight_decay = 0.0
    momentum = None
    enable_lr_schedule = True
    decay_step = 10
    gamma = 0.1
    metric_ks = [5, 10, 20]  # Evaluation metrics at top 5, 10, and 20
    best_metric = 'NDCG@10'  # Specify the metric used to determine the best model
    log_period_as_iter = 100  # Log metrics and losses every 100 iterations





    
# Instantiate the configuration
args = Config()

def main():
    # Initialize dataset and dataloaders
    dataset = UserInteractionDataset(
        args.dataset_path,
        args.bert_max_len,
        args.mask_prob,
        args.mask_token
    )
    
    # Ideally, you should have separate datasets for train, validation, and test
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers
    )

    # Ensure the export root exists
    if not os.path.exists(args.export_root):
        os.makedirs(args.export_root)
    
    # Initialize the BERT model
    bert_model = BERTModel(args)
    bert_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the trainer with additional loaders and export root
    trainer = BERTTrainer(
        args=args,
        model=bert_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        export_root=args.export_root
    )

    # Training loop
    for epoch in range(args.num_epochs):
        trainer.train(epoch)

    # Optionally, start the evaluation on test data
    trainer.test()

if __name__ == "__main__":
    main()
