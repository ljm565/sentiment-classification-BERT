# Data Preparation
여기서는 기본적으로 [Google Play Store App review](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/) 데이터셋을 활용하여 BERT 감성 분류 모델 학습 튜토리얼을 진행합니다.
Custom 데이터를 이용하기 위해서는 아래 설명을 참고하시기 바랍니다.

### 1. Google Store Review
Google Store Review 데이터를 학습하고싶다면 아래처럼 `config/config.yaml`의 `google_store_review_train`을 `True`로 설정하면 됩니다.
```yaml
google_store_review_train: True
google_store_review:
    path: data/
    trainset_prop: 0.8
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
만약 custom 데이터를 학습하고 싶다면 아래처럼 `config/config.yaml`의 `google_store_review_train`을 `False`로 설정하면 됩니다.
다만 `src/utils/data_utils.py`에 custom dataloader를 구현해야할 수 있습니다.
```yaml
google_store_review_train: False
google_store_review:
    path: data/
    trainset_prop: 0.8
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```