import pandas as pd

# Try using a different delimiter
df = pd.read_csv(r"/content/001ssb.txt", sep='\t')  # Try tab delimiter
# df = pd.read_csv(r"/content/001ssb.txt", sep=' ')  # Try space delimiter

df.head()


!pip install gensim

import gensim
import os

import nltk
nltk.download('punkt_tab')

from nltk import sent_tokenize
from gensim.utils import simple_preprocess

story = []

# Open the file directly
with open('001ssb.txt', 'r') as f:
    corpus = f.read()
    raw_sent = sent_tokenize(corpus)
    for sent in raw_sent:
        story.append(simple_preprocess(sent))


model = gensim.models.Word2Vec(
    window=10,
    min_count=2
)        


model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)



#For finding the dose not mactching words 
model.wv.doesnt_match(['king', 'queen', 'man', 'woman','monkey'])
#output ----- Donkey

model.wv.most_similar("donkey")