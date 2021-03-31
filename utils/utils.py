import yaml
import pickle


def load_cfg(yaml_file_path):
    """
    Loads a yaml config file
    :param yaml_file_path: path of yaml file
    :return: config corresponding the path
    """
    with open(yaml_file_path, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    return cfg


def load_pickle(path):
    """
    Loads a pickle file
    :param path: path of pickle file
    :return: pickle file data
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def save_pickle(path, data):
    """
    Save data in pickle file
    :param path: path to save
    :param data: dat that to be saved
    :return: None
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        f.close()
