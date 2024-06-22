# Training BERT Sentiment Classification
Here, we provide guides for training a BERT sentiment classification model.

### 1. Configuration Preparation
To train a BERT sentiment classification model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu                           # examples: [0], [0,1], [1,2,3], cpu, mps... 

# project config
project: outputs/bert_classifier
name: google_store_review

# model config
pretrained_model: bert-base-uncased   # [bert-base-uncased, bert-base-cased, bert-large-uncased, ...] Hugging Face's pre-trained BERT model path
max_len: 512
class_num: 3                          # The number of classes of the BERT classifier

# data config
workers: 0                            # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
google_store_review_train: True       # if True, Google store reivew data will be loaded automatically.
google_store_review:
    path: data/
    trainset_prop: 0.8                # The ratio of the data used for training. If it is set to 0.8, then the validation and test data will each be set to a ratio of 0.1.
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
batch_size: 32
steps: 64000
warmup_steps: 100
lr0: 0.001
lrf: 0.001                            # last_lr = lr0 * lrf
scheduler_type: 'cosine'              # ['linear', 'cosine']
patience: 5                           # Early stopping epochs. Epochs are automatically calculated according to current steps.

# logging config
common: ['train_loss', 'train_acc', 'validation_loss', 'validation_acc', 'lr']
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-p`, `--port`]: (default: `10001`) NCCL port for DDP training.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```
When training started, the learning rate curve will be saved in `${project}/${name}/vis_outputs/lr_schedule.png` automatically based on the values set in `config/config.yaml`.
When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.