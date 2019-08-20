
from imports import *

ap = argparse.ArgumentParser(description="Train a sentiment analysis model in the Imdb data set")
ap.add_argument("-data_path", "--data_path", type = str, help = "path to input data")
ap.add_argument("-bs", "--bs", type = float, default = 16, help = "batch size")
ap.add_argument("-epochs", "--epochs", type = float, default = 10, help = "number of epochs")
ap.add_argument("-emb_dim", "--emb_dim", type = int, default = 200, help = "Embedding dimension")
ap.add_argument("-hid_dim", "--hid_dim", type = int, default = 50, help = "Hidden size")
# ap.add_argument("-load", "--load_model", type = float, default = 0, help = "load model bool")
ap.add_argument("-device", "--device", type = str, default = 'cpu', help = "device")
ap.add_argument("-max_sent", "--max_sent", type = int, default = 148, help = "max number of sentences to be considered together")
ap.add_argument("-max_sent_words", "--max_sent_words", type = int, default = 2802, help = "max number of words to be considered in a sentence")
kvargs = vars(ap.parse_args())

print()
print('+------------------------------------+')
print('|              PRODUVIA              |')
print('+------------------------------------+')
print()

print(kvargs)

PATH = Path(kvargs['data_path'])



# train_ds = doc_s_w(train_df)
# val_ds = doc_s_w(val_df)
# test_ds = doc_s_w(test_df)

# train_dl = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=dynamic_word_sentece_padding, drop_last=True)
# val_dl = DataLoader(val_ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dynamic_word_sentece_padding, drop_last=True)
# test_dl = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dynamic_word_sentece_padding, drop_last=True)

# # model
# model = HiererchicalAttention(embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, vocab_size=vocab_size, 
#                               bs=BATCH_SIZE, out_size=OUT_SIZE, dropout=.3).cuda()
# h_0 = model.initHidden(cuda=True)

# # train
# train(n_epochs, train_dl, model, h_0, val_dl, max_lr=5e-3)