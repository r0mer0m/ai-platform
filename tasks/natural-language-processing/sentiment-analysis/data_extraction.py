# methods for dataset-specific data extraction

from imports import *

# data frame
def get_paths_labels(data_path:Path) -> (list,list):
    '''
    Creates two lists w\ the paths and the sentiment of the reviews. 

    Assumes aclImdb folder-structure.

    :param data_path: either path to training folder or to testing folder.
    :return: tuple with two lists: paths to text and sentiment. 
    E.g.:

        train_path = Path('../aclImdb/train')
        paths, y = get_paths_labels()

    '''

    pos_rev = list((data_path/'pos').iterdir())
    N_pos = len(pos_rev)

    neg_rev = list((data_path/'neg').iterdir())
    N_neg = len(neg_rev)

    return pos_rev.extend(neg_rev), [1]*N_pos + [0]*N_neg
