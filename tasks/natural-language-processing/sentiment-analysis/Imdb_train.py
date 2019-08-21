
from imports import *
from utils import imdb_extraction, train_val_split
from data_serving import ha_dataset, dynamic_ws_padding
from text_processing import create_vocab
from models import HiererchicalAttention

# parameters
ap = argparse.ArgumentParser(description="Train a sentiment analysis model in the Imdb data set")
ap.add_argument("-data_path", "--data_path", type = str, help = "path to input data")
ap.add_argument("-bs", "--bs", type = float, default = 16, help = "batch size")
ap.add_argument("-epochs", "--epochs", type = float, default = 6, help = "number of epochs")
ap.add_argument("-emb_dim", "--emb_dim", type = int, default = 200, help = "Embedding dimension")
ap.add_argument("-hid_dim", "--hid_dim", type = int, default = 50, help = "Hidden size")
# ap.add_argument("-load", "--load_model", type = float, default = 0, help = "load model bool")
ap.add_argument("-device", "--device", type = str, default = 'cpu', help = "device")
ap.add_argument("-drop_out", "--drop_out", type = float, default = 0.0, help = "drop out percentage")
ap.add_argument("-max_sent", "--max_sent", type = int, default = 148, help = "max number of sentences to be considered together")
ap.add_argument("-max_sent_words", "--max_sent_words", type = int, default = 2802, help = "max number of words to be considered in a sentence")
ap.add_argument("-min_appearence", "--min_appearence", type = int, default = 5, help = "min number of appearences to include a token in vocab")
kvargs = vars(ap.parse_args())

print()
print('+------------------------------------+')
print('|              PRODUVIA              |')
print('+------------------------------------+')
print()

PATH = Path(kvargs['data_path'])
DEVICE = kvargs['device']
padding = partial(dynamic_ws_padding, max_sent=kvargs['max_sent'], max_words=kvargs['max_sent_words'])

# create mapping text <-> sent
data = imdb_extraction(PATH/'train')
test = imdb_extraction(PATH/'test')

# split data into train & validation
train, valid = train_val_split(data, train_ptg=.8)

# create vocabulary
t2i, i2t = create_vocab(train)

# create datasets
train_ds = ha_dataset(train, t2i)
valid_ds = ha_dataset(valid, t2i)

# create dataloaders
train_dl = DataLoader(train_ds, shuffle=True, batch_size=kvargs['bs'], 
                    collate_fn=padding, drop_last=True)
valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=kvargs['bs'], 
                    collate_fn=padding, drop_last=True)

# define architecutre
model = HiererchicalAttention(embedding_dim=kvargs['emb_dim'], hidden_size=kvargs['hid_dim'], vocab_size=len(t2i), 
                              bs=kvargs['bs'], out_size=1, max_words=kvargs['max_sent_words'], max_sentences=kvargs['max_sent'],
                              dropout=kvargs['drop_out']).to(DEVICE)

h_0 = model.initHidden(DEVICE)

# train(kvargs['epochs'], train_dl, model, h_0, valid_dl, max_lr=5e-3)

print('<----- END ----->')
