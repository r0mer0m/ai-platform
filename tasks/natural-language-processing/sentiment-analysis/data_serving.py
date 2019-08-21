from imports import *
from text_processing import create_vocab, spacy_tok, split_sentece

class ha_dataset(Dataset):
    
    def __init__(self, data, vocab2index, unk_idx=1, encoding='utf-8'):
        
        # df = order_docs(df)
        self.data = data
        self.unk_idx = unk_idx
        self.vocab2index = vocab2index
        self.encoding = encoding
        
    def __len__(self,):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        file = self.data[idx][0]
        doc_sent = split_sentece(file.read_text(self.encoding))
        doc_sent_tok = [spacy_tok(sentence) for sentence in doc_sent]
        
        x = [[self.vocab2index.get(w, self.unk_idx) for w in s] for s in doc_sent_tok]
        
        n_sent = len(x)
        max_n_words = max([len(s) for s in x])
        
        return x, self.data[idx][1], n_sent, max_n_words


def dynamic_ws_padding(max_sent:int, max_words:int, batch:tuple):
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

