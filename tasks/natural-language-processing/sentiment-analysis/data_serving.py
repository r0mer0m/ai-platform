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