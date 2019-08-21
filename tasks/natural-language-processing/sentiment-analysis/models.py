from imports import *


class Attention(nn.Module):
    
    def __init__(self,  input_size, dropout=0):
        super().__init__()
        
        self.linear1 = nn.Linear(input_size, input_size)
                
        self.linear2 = nn.Linear(input_size, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x):
        
        out = F.tanh(self.linear1(x))
        
        out = self.linear2(out)

        out = out.squeeze()
        
        out = self.dropout(out)

        w = F.softmax(out, dim=-1)
                
        out = torch.bmm(w.unsqueeze(-1).permute(0,2,1), x).squeeze()

        return out


class HiererchicalAttention(nn.Module):
    
    def __init__(self, embedding_dim, hidden_size, vocab_size, bs, out_size,
                max_words, max_sentences, dropout=0):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bs = bs

        # word embbeding
        self.emb = nn.Embedding(num_embeddings=vocab_size, 
                                embedding_dim=embedding_dim)
        
        # word level
        self.word_GRU = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                               bidirectional=True, batch_first=True, dropout=dropout)
    
        self.word_Attention = Attention(2*hidden_size, dropout=dropout)
        
        # sentece level
        self.sentence_GRU = nn.GRU(input_size=2*hidden_size, hidden_size=hidden_size,
                               bidirectional=True, batch_first=True, dropout=dropout)
        
        self.sentence_Attention = Attention(2*hidden_size, dropout=dropout)
        
        # final linear
        self.linear = nn.Linear(2*hidden_size, out_size)
        
    def forward(self, x, h_0):
        
        sent_encoding  = []
        
        # GRU + Attention (word - level) - :output: vector encoding the sentence 
        for sent_idx in range(x.shape[1]):
            
            x_i = x[:,sent_idx,:]
            
            out_i = self.emb(x_i)
            
            out_i = self.word_GRU(out_i, h_0)[0]
            
            out_i = self.word_Attention(out_i)

            sent_encoding.append(out_i.unsqueeze(-2))
        
        out = torch.cat(sent_encoding, dim=-2)
        
        # GRU + Attention (sentence - level) - :output: vector encoding the sentence 
        # We don't want sentence level attention if there is only one sentence (this just adds noise)
        if x.shape[1]>1:
            
            out = self.sentence_GRU(out, h_0)[0]

            out = self.sentence_Attention(out)
        
        out = self.linear(out)
        
        return out.squeeze()
    
    def initHidden(self, device):
        hidden=torch.zeros(2, self.bs, self.hidden_size, requires_grad=False)
        return hidden.to(device)