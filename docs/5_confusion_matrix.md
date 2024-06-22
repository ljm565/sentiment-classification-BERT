# Confusion Matrix Visualization
Here, we provide a guide for visualizing a confusion matrix of a trained BERT sentiment classification model.

### 1. Confusion Matrix Visualization
#### 1.1 Arguments
There are several arguments for running `src/run/vis_statistics.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to visualize confusion matrix. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to visualize.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-d`, `--dataset_type`]: (default: `validation`) Choose one of [`train`, `validation`, `test`].


#### 1.2 Command
`src/run/vis_statistics.py` file is used to visualize confusion matrix of the model with the following command:
```bash
python3 src/run/vis_statistics.py --resume_model_dir ${project}/${name}
```