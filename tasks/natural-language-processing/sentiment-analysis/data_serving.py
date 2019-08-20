from imports import *
from text_processing import create_vocab, spacy_tok, split_sentece

class doc_s_w(Dataset):
    
    def __init__(self, df, vocab2index, def_idx=1):
        
        # df = order_docs(df)
        self.paths = df.file_paths
        self.y = df.ratings.values - 1
        self.def_idx = def_idx
        self.vocab2index = vocab2index
        
    def __len__(self,):
        return len(self.y)
        
    def __getitem__(self, idx):
        
        file = self.paths[idx]
        doc_sent = split_sentece(file.read_text('utf-8'))
        doc_sent_tok = [spacy_tok(sentence) for sentence in doc_sent]
        
        x = [[self.vocab2index.get(w, self.def_idx) for w in s] for s in doc_sent_tok]
        
        max_n_words = max([len(s) for s in x])
        
        return x, self.y[idx], len(x), max_n_words


def dynamic_word_sentece_padding(batch, max_sent, max_words):
    '''
    Batch-level dynamic padding.
    ''' 
    
#     compute dimensions of batch
    dim_sent = min(max_sent, max([b[2] for b in batch]))
    dim_words = min(max_words, max([b[3] for b in batch]))
    
#     Create sentece input
    X =[]
    for sentences,*_ in batch:
        A = np.zeros([dim_sent, dim_words])
        for i in range(min([len(sentences),dim_sent])):
            fill_up_to = min(len(sentences[i]), dim_words)
            A[i,:fill_up_to] = sentences[i][:fill_up_to]
        X.append(A)
            
    y = [b[1] for b in batch]

    new_batches = list(zip(X,y))
    return default_collate(new_batches)

