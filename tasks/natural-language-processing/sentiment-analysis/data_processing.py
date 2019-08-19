
from imports import *

# data frame
def construct_df(data_path:Path):
    '''
    Creates a panda data frame with the `file_paths` and it's respective `ratings`.

    Assumes aclImdb structure.

    :param data_path: either path to training folder or to testing folder.
    :return: pd.DataFrame
    '''
    regex = re.compile(r'[\d]*_(\d+)\.txt')
    file_paths = []
    ratings = []
    for sent_type in ['pos','neg']:
        for file in (data_path/sent_type).iterdir():
            file_paths.append(file)
            ratings.append(regex.sub( r"\1", str(file).split('/')[-1]))

    return pd.DataFrame(list(zip(file_paths,ratings)), columns=['file_paths', 'ratings'])

