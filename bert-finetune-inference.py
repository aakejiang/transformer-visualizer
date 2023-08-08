dev_mode = False

# 1. Some basic setting
from pathlib import Path
from typing import Callable, Dict

pretrained_model_name_or_path = 'bert-base-uncased'
task_name = 'mnli'
experiment_id = 'pruning_bert_mnli'

# heads_num and layers_num should align with pretrained_model_name_or_path
heads_num = 12
layers_num = 12

# used to save the experiment log
log_dir = Path(f'./pruning_log/{pretrained_model_name_or_path}/{task_name}/{experiment_id}')
log_dir.mkdir(parents=True, exist_ok=True)

# used to save the finetuned model and share between different experiemnts with same pretrained_model_name_or_path and task_name
model_dir = Path(f'./models/{pretrained_model_name_or_path}/{task_name}')
model_dir.mkdir(parents=True, exist_ok=True)

# used to save GLUE data
data_dir = Path(f'./data')
data_dir.mkdir(parents=True, exist_ok=True)

# set seed
from transformers import set_seed
set_seed(1024)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Create dataloaders

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding

task_to_keys = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

def prepare_dataloaders(cache_dir=data_dir, train_batch_size=32, eval_batch_size=32):
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    data_collator = DataCollatorWithPadding(tokenizer)

    # used to preprocess the raw data
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if 'label' in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result['labels'] = examples['label']
        return result

    raw_datasets = load_dataset('glue', task_name, cache_dir=cache_dir)
    for key in list(raw_datasets.keys()):
        if 'test' in key:
            raw_datasets.pop(key)

    processed_datasets = raw_datasets.map(preprocess_function, batched=True,
                                          remove_columns=raw_datasets['train'].column_names)

    train_dataset = processed_datasets['train']
    if task_name == 'mnli':
        validation_datasets = {
            'validation_matched': processed_datasets['validation_matched'],
            'validation_mismatched': processed_datasets['validation_mismatched']
        }
    else:
        validation_datasets = {
            'validation': processed_datasets['validation']
        }

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size)
    validation_dataloaders = {
        val_name: DataLoader(val_dataset, collate_fn=data_collator, batch_size=eval_batch_size) \
            for val_name, val_dataset in validation_datasets.items()
    }

    return train_dataloader, validation_dataloaders


train_dataloader, validation_dataloaders = prepare_dataloaders()

# 3. Training function & evaluation function.

import functools
import time

import torch.nn.functional as F
import evaluate
from transformers.modeling_outputs import SequenceClassifierOutput


def training(model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
             max_steps: int = None,
             max_epochs: int = None,
             train_dataloader: DataLoader = None,
             distillation: bool = False,
             teacher_model: torch.nn.Module = None,
             distil_func: Callable = None,
             log_path: str = Path(log_dir) / 'training.log',
             save_best_model: bool = False,
             save_path: str = None,
             evaluation_func: Callable = None,
             eval_per_steps: int = 1000,
             device=None):

    assert train_dataloader is not None

    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    current_step = 0
    best_result = 0

    total_epochs = max_steps // len(train_dataloader) + 1 if max_steps else max_epochs if max_epochs else 3
    total_steps = max_steps if max_steps else total_epochs * len(train_dataloader)

    print(f'Training {total_epochs} epochs, {total_steps} steps...')

    for current_epoch in range(total_epochs):
        for batch in train_dataloader:
            if current_step >= total_steps:
                return
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss

            if distillation:
                assert teacher_model is not None
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                distil_loss = distil_func(outputs, teacher_outputs)
                loss = 0.1 * loss + 0.9 * distil_loss

            loss = criterion(loss, None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # per step schedule
            if lr_scheduler:
                lr_scheduler.step()

            current_step += 1

            if current_step % eval_per_steps == 0 or current_step % len(train_dataloader) == 0:
                result = evaluation_func(model) if evaluation_func else None
                with (log_path).open('a+') as f:
                    msg = '[{}] Epoch {}, Step {}: {}\n'.format(time.asctime(time.localtime(time.time())), current_epoch, current_step, result)
                    f.write(msg)
                # if it's the best model, save it.
                if save_best_model and (result is None or best_result < result['default']):
                    assert save_path is not None
                    torch.save(model.state_dict(), save_path)
                    best_result = None if result is None else result['default']


def distil_loss_func(stu_outputs: SequenceClassifierOutput, tea_outputs: SequenceClassifierOutput, encoder_layer_idxs=[]):
    encoder_hidden_state_loss = []
    for i, idx in enumerate(encoder_layer_idxs[:-1]):
        encoder_hidden_state_loss.append(F.mse_loss(stu_outputs.hidden_states[i], tea_outputs.hidden_states[idx]))
    logits_loss = F.kl_div(F.log_softmax(stu_outputs.logits / 2, dim=-1), F.softmax(tea_outputs.logits / 2, dim=-1), reduction='batchmean') * (2 ** 2)

    distil_loss = 0
    for loss in encoder_hidden_state_loss:
        distil_loss += loss
    distil_loss += logits_loss
    return distil_loss


def evaluation(model: torch.nn.Module, validation_dataloaders: Dict[str, DataLoader] = None, device=None):
    assert validation_dataloaders is not None
    training = model.training
    model.eval()

    is_regression = task_name == 'stsb'
    metric = evaluate.load('glue', task_name)

    result = {}
    default_result = 0
    for val_name, validation_dataloader in validation_dataloaders.items():
        for batch in validation_dataloader:
            batch.to(device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=predictions,
                references=batch['labels'],
            )
        result[val_name] = metric.compute()
        default_result += result[val_name].get('f1', result[val_name].get('accuracy', 0))
    result['default'] = default_result / len(result)

    model.train(training)
    return result


evaluation_func = functools.partial(evaluation, validation_dataloaders=validation_dataloaders, device=device)


def fake_criterion(loss, _):
    return loss

# 4. Prepare pre-trained model and finetuning on downstream task.
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertForSequenceClassification


def create_pretrained_model():
    is_regression = task_name == 'stsb'
    num_labels = 1 if is_regression else (3 if task_name == 'mnli' else 2)
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
    model.bert.config.output_hidden_states = True
    return model


# fine-tune bert if not exists, else reloading the fine-tuned model.
def create_finetuned_model():
    finetuned_model = create_pretrained_model()
    finetuned_model_state_path = Path(model_dir) / 'finetuned_model_state.pth'

    if finetuned_model_state_path.exists():
        print("================= fine-tuned model exists =================")
        finetuned_model.load_state_dict(torch.load(finetuned_model_state_path, map_location='cpu'))
        finetuned_model.to(device)
    elif dev_mode:
        pass
    else:
        steps_per_epoch = len(train_dataloader)
        training_epochs = 3
        optimizer = Adam(finetuned_model.parameters(), lr=3e-5, eps=1e-8)

        def lr_lambda(current_step: int):
            return max(0.0, float(training_epochs * steps_per_epoch - current_step) / float(training_epochs * steps_per_epoch))

        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        training(finetuned_model, optimizer, fake_criterion, lr_scheduler=lr_scheduler,
                 max_epochs=training_epochs, train_dataloader=train_dataloader, log_path=log_dir / 'finetuning_on_downstream.log',
                 save_best_model=True, save_path=finetuned_model_state_path, evaluation_func=evaluation_func, device=device)
    return finetuned_model


finetuned_model = create_finetuned_model()

res = evaluation(finetuned_model, validation_dataloaders, device)
print(res)