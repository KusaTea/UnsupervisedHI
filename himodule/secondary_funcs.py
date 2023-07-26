import pickle
import os

def save_object(obj: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def check_path(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)