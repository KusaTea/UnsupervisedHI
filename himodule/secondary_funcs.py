import pickle
import os

def save_object(obj: object, path: str):
    '''Saves any objets to .pkl format file.'''
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path: str):
    '''Loads objets from .pkl format file.'''
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def check_path(pth):
    '''Check if the path exists, otherwise creates the path.'''
    if not os.path.exists(pth):
        os.makedirs(pth)