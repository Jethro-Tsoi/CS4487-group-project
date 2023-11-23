import os
import torch
from transformers import ViTImageProcessor, ViTImageProcessor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the processor
model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)


train_dataset = ImageFolder('project_data/train', transform=ToTensor())
validation_dataset = ImageFolder('project_data/val', transform=ToTensor())

print(train_dataset[0])

example = processor(
    train_dataset[0][0],
    return_tensors='pt'
)
print(example)
print(example['pixel_values'].shape)

import torch

# device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def preprocess(batch):
    # take a list of PIL images and turn them into pixel values
    inputs = processor(
        batch[0],  # Assuming batch['img'] is the PIL image
        return_tensors='pt'
    )
    # include the labels
    inputs['label'] = torch.tensor([batch[1]])  # Assuming batch['label'] is the label
    return inputs

# open the preprocessed datasets if preprocess/prepared_train.pt exist
prepared_train = None
prepared_test = None
if os.path.exists('preprocess/prepared_train.pt'):
    prepared_train = torch.load('preprocess/prepared_train.pt')
    prepared_test = torch.load('preprocess/prepared_test.pt')
else:
    # Apply preprocess to the datasets
    prepared_train = [preprocess(train_dataset[i]) for i in range(len(train_dataset))]
    prepared_test = [preprocess(validation_dataset[i]) for i in range(len(validation_dataset))]

    # Save the preprocessed datasets locally
    torch.save(prepared_train, 'preprocess/prepared_train.pt')
    torch.save(prepared_test, 'preprocess/prepared_test.pt')


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

import numpy as np
from datasets import load_metric

# accuracy metric
metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids
    )

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./cifar",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)

from transformers import ViTForImageClassification

labels = train_dataset.classes
# print(labels)

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,  # classification head
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)


model.to(device)

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./vit-base-beans",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_train,
    eval_dataset=prepared_test,
    tokenizer=processor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_test)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
