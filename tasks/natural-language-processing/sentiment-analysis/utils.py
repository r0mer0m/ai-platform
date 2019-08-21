# methods for dataset-specific data extraction

from imports import *

###################
# TRAIN/VAL SPLIT
###################

def train_val_split(data:list, train_ptg:int=0.8) -> (list, list):
    '''
    Given a list [(x1,y1),(x2,y2),...] splits it into two at random based on `train_ptg`

    :param data: [(x1,y1),(x2,y2),...]
    :param train_ptg: rough percentage of the data to use in training. 
    :return: 2 lists with the input format. First for training, second for validation. 
    '''
    tr_idx = np.random.rand(len(data)) < train_ptg
    return data[tr_idx], data[~tr_idx]



###################
# DATA EXTRACTION
###################

# data frame
def imdb_extraction(data_path:Path) -> list:
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

    return np.array(list(zip(pos_rev + neg_rev, [1]*N_pos + [0]*N_neg)))
