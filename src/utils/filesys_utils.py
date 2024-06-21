import os
import pickle
import pandas as pd
from pathlib import Path

from utils import LOGGER, colorstr
from utils.func_utils import labeling



def read_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_dataset(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def make_project_dir(config, is_rank_zero=False):
    """
    Make project folder.

    Args:
        config: yaml config.
        is_rank_zero (bool): make folder only at the zero-rank device.

    Returns:
        (path): project folder path.
    """
    prefix = colorstr('make project folder')
    project = config.project
    name = config.name

    save_dir = os.path.join(project, name)
    if os.path.exists(save_dir):
        if is_rank_zero:
            LOGGER.info(f'{prefix}: Project {save_dir} already exists. New folder will be created.')
        save_dir = os.path.join(project, name + str(len(os.listdir(project))+1))
    
    if is_rank_zero:
        os.makedirs(project, exist_ok=True)
        os.makedirs(save_dir)
    
    return Path(save_dir)


def yaml_save(file='data.yaml', data=None, header=''):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    save_path = Path(file)
    print(data.dumps())
    with open(save_path, "w") as f:
        f.write(data.dumps(modified_color=None, quote_str=True))
        LOGGER.info(f"Config is saved at {save_path}")


def preprocess_data(path):
    write_path = os.path.join(path, 'processed/data.pkl')
    if not (os.path.isfile(os.path.join(path, 'processed/data.pkl'))):
        LOGGER.info('Processing the Google Play Store Apps review data')
        
        df = pd.read_csv(os.path.join(path, 'raw/reviews.csv'))
        content, score = df.loc[:, 'content'], labeling(df.loc[:, 'score'].tolist())
        dataset = [(c, s) for c, s in zip(content, score)]
        
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    return write_path