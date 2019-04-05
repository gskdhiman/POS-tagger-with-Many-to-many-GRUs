# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:11:15 2019

@author: GU389021
"""
import os 
import nltk
import numpy as np
from sklearn.model_selection import train_test_split 

word_emb_path = os.path.join('..','Word_embeddings','glove.6B.50d.txt')  
tagged_sentences = nltk.corpus.treebank.tagged_sents()
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))
  
sentences, sentence_tags =[], [] 
for tagged_sentence in tagged_sentences:
    sentence, tags = zip(*tagged_sentence)
    sentences.append(sentence)
    sentence_tags.append(tags)
  
print(sentences[5])
print(sentence_tags[5])

train_sentences,test_sentences,train_tags,test_tags = train_test_split(sentences, 
                                                                            sentence_tags, 
                                                                            test_size=0.2)
tags =  set([]) 
for ts in train_tags:
    for t in ts:
        tags.add(t)
del t     

from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
GloveEmbeddings,emb_dim = {}, 50
def loadEmbeddings(Glove_embeddingFileName):
    global GloveEmbeddings,emb_dim
    print("Starting loading embeddings")
    fe = open(Glove_embeddingFileName,"r",encoding="utf-8",errors="ignore")
    for line in tqdm(fe):
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        GloveEmbeddings[word]=np.array(vec).astype('float32')
    GloveEmbeddings["zerovec"] = np.zeros(emb_dim).astype('float32')
    fe.close()
    print("Embeddings Loaded")
    
loadEmbeddings(word_emb_path)    
seq_len =30    
def getembedding(x):
    global GloveEmbeddings
    emb = [GloveEmbeddings[word] if word in GloveEmbeddings.keys() else GloveEmbeddings['zerovec'] for word in x]
    return pad_sequences([emb],maxlen=seq_len, dtype='float32',padding = 'post')


tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding


train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
 
for s in train_sentences:
    emb = []
    try:
        emb.append(getembedding(s))
    except KeyError:
        continue
    train_sentences_X.append(emb)

x_train = np.array(train_sentences_X).reshape(3131,30,50)    

for s in test_sentences:
    emb = []
    try:
        emb.append(getembedding(s))
    except KeyError:
        continue
    test_sentences_X.append(emb)
    
x_test = np.array(test_sentences_X).reshape(783,30,50)

for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])
 
for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])

print(x_train[0]) 
print(x_test[0])
print(train_tags_y[0])
print(test_tags_y[0])

MAX_LENGTH = len(max(test_tags_y, key=len))
print(MAX_LENGTH)  


 
y_train = pad_sequences(train_tags_y, maxlen=30, padding='post')
y_test= pad_sequences(test_tags_y, maxlen=30, padding='post') 

print(x_train[0]) 
print(x_test[0])
print(y_train[0])
print(y_test[0])

from keras import backend as K
def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy
    
 
from keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed,Dropout
from keras.models import Model
inp = Input(shape = (30,50))
out = Bidirectional(LSTM(256, return_sequences=True))(inp)
out = Dropout(0.7)(out)
out = Bidirectional(LSTM(128, return_sequences=True))(out)
out = Dropout(0.5)(out)
out = TimeDistributed(Dense(len(tag2index),activation = 'softmax'))(out)
model = Model(inputs = inp , outputs = out) 
model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
                   metrics=['accuracy', ignore_class_accuracy(0)])
 
model.summary()

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

cat_train_tags_y = to_categorical(y_train, len(tag2index))
print(cat_train_tags_y[0])


model_history = model.fit(x_train, cat_train_tags_y, batch_size=128, epochs=20, validation_split=0.2,verbose =1)

model.save('pos_tagger_model.h5') 

predictions = model.predict(x_test)
print(predictions.shape)


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
        token_sequences.append(token_sequence)
    return token_sequences

print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})) 

 
