# Data Preparation
Here, we will proceed with a BERT sentiment classification model training tutorial using the [Google Play Store App review](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/) dataset by default.
Please refer to the following instructions to utilize custom datasets.

### 1. Google Store Review
If you want to train on the Google Store Review dataset, simply set the `google_store_review_train` value in the `config/config.yaml` file to `True` as follows.
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
If you want to train your custom dataset, set the `google_store_review_train` value in the `config/config.yaml` file to `False` as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
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
