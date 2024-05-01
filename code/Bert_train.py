#%%
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback, DataCollatorWithPadding
from datasets import load_from_disk, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import cohen_kappa_score, f1_score
import os
from accelerate import Accelerator, DataLoaderConfiguration

#%%
df = pd.read_parquet('../Code/data/final_df.parquet')
print('shape:', df.shape)
# Encode the category
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['categories'])
# Save the label_encoder
joblib.dump(label_encoder, 'label_encoder.joblib')


# Split into train, test, and validation and save files (0.8, 0.1, 0.1)
train_df, temp_test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, valid_df = train_test_split(temp_test_df, test_size=0.5, random_state=42)

#%%
# Convert dataframes to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

print(f'Train shape {train_df.shape}')
print(f'Test shape {test_df.shape}')
print(f'Validation shape {valid_df.shape}')
'''
Train shape (46559, 7)
Test shape (5820, 7)
Validation shape (5820, 7)
'''

#%%
''' Tokenize with BERT '''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(dataset):
   return tokenizer(dataset['review_text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

#%%
'''Save dataset to disk '''
data_path = './Code/train_data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

train_dataset.save_to_disk('./Code/train_data/train_dataset')
val_dataset.save_to_disk('./Code/train_data/val_dataset')
test_dataset.save_to_disk('./Code/train_data/test_dataset')

#%%
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    kappa = cohen_kappa_score(labels, predictions)

    return {
        'f1_ave': f1_weighted,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'cohen_kappa': kappa
    }

class metric_print(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print("\nMetrics results at Epoch {}: {}".format(state.epoch, metrics))


#%%
'''Load the dataset'''
train_dataset = load_from_disk('../Code/train_data/train_dataset')
val_dataset = load_from_disk('../Code/train_data/val_dataset')
test_dataset = load_from_disk('../Code/train_data/test_dataset')

# Set the format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

#Load BERT and Add a Classification Head
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Dataloader configuration
dataloader_config = DataLoaderConfiguration(
    dispatch_batches=None,
    split_batches=False
)

accelerator = Accelerator(
    dataloader_config=dataloader_config
)




'''*************************************Training*************************************'''
# Train and fine-tuning
training_args = TrainingArguments(
    output_dir='../Code/result_2',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-4,
    per_device_train_batch_size=30,
    per_device_eval_batch_size=30,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1_micro'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), metric_print()]
)


trainer.train()



'''*************************************TEST*************************************'''
#%%
model_path = "../Code/result/checkpoint-4656"
model = BertForSequenceClassification.from_pretrained(model_path)

training_args = TrainingArguments(
    output_dir='../Code/result_2',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-4,
    per_device_train_batch_size=30,
    per_device_eval_batch_size=30,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1_micro'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), metric_print()]
)

results = trainer.evaluate()
print("test results:", results)

#%%
'''Predict category based on text'''
def model_predict(path, text):
    model = BertForSequenceClassification.from_pretrained(path)
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)

    logits = output.logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    label_encoder = joblib.load('label_encoder.joblib')
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])

    return predicted_class[0]
