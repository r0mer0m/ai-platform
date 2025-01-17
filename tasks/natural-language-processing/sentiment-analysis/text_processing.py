from imports import *

_re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
_spacy_tokenizer = spacy.load('en')

def sub_br(x:str) -> str:
    ''' 
    Replaces newline in HTML format (<br>) by \n in the text
    
    :param x: input string
    :return: output string
    '''
    return _re_br.sub("\n", x.lower())

def split_sentece(x:str, splt_char: list or tuple =['.','!','?','\\n']) -> list: 
    ''' 
    Splits document into sentences using the characters specified in the `splt_char` argument.
    
    :param x: doc (str) to be splitted into sentences
    :param splt_char: list of characters (str's) using for splitting
    :return: list of sentences (str's)
    '''
    special_char = {'.','!','?','-',']','[',')','(','\\n'}
    expression = ' |'.join(['\\'+c if c in special_char else c for c in splt_char])
    return re.split(expression, sub_br(x))

def spacy_tok(x: str) -> list: 
    '''
    Splits a sentence using Spacy tokenizer after applying `sub_br`.
    
    :param x: sentence to be tokenized
    :return: list of tokens 
    '''
    return [tok.text for tok in _spacy_tokenizer.tokenizer(sub_br(x))]

def create_vocab(data:iter, encoding:str='utf-8', minimum:int=5) -> (dict,list):
    '''
    Creates the vocabulary using the provided `paths` to the documents.
    
    :param paths: iterative of the paths to the files in pathlib.PosixPath format.
    :param encoding: text format.
    :param minimum: Minimum (not included) occurrences for a token to be considered.
    :return: token 2 index mapping (dict),  index 2 token mapping (list)
    '''

    counts = Counter()
    print('Creating vocabulary ...')
    for path in tqdm(data[:,0]): 
        counts.update(spacy_tok(path.read_text(encoding=encoding)))

    # create mappings
    t2i = {"<PAD>":0, "<UNK>":1}
    i2t = ["<PAD>", "<UNK>"]
    for word in counts:
        if counts[word]>minimum:
            t2i[word] = len(i2t)
            i2t.append(word)
    
    return t2i, i2t