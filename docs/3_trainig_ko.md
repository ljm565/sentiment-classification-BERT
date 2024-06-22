# Training BERT Sentiment Classification
여기서는 BERT 감성 분류 모델을 학습하는 가이드를 제공합니다.

### 1. Configuration Preparation
BERT 감성 분류 모델을 학습하기 위해서는 Configuration을 작성하여야 합니다.
Configuration에 대한 option들의 자세한 설명 및 예시는 다음과 같습니다.

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
`src/run/train.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [`-c`, `--config`]: 학습 수행을 위한 config file 경로.
* [`-m`, `--mode`]: [`train`, `resume`] 중 하나를 선택.
* [`-r`, `--resume_model_dir`]: mode가 `resume`일 때 모델 경로. `${project}/${name}`까지의 경로만 입력하면, 자동으로 `${project}/${name}/weights/`의 모델을 선택하여 resume을 수행.
* [`-l`, `--load_model_type`]: [`metric`, `loss`, `last`] 중 하나를 선택.
    * `metric`(default): Valdiation accuracy가 최대일 때 모델을 resume.
    * `loss`: Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.
* [`-p`, `--port`]: (default: `10001`) DDP 학습 시 NCCL port.


#### 2.2 Command
`src/run/train.py` 파일로 다음과 같은 명령어를 통해 BERT 모델을 학습합니다.
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```
학습이 시작되면 예상 학습 learning curve 그래프가 `${project}/${name}/vis_outputs/lr_schedule.png` 경로에 저장됩니다.
모델 학습이 끝나면 `${project}/${name}/weights`에 체크포인트가 저장되며, `${project}/${name}/args.yaml`에 학습 config가 저장됩니다.