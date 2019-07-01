import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
import re
import string
from time import time  # To time our operations
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec



# Function to remove Stopwords
def remove_stopwords(line):
    stop_words = set(stopwords.words('english'))
    line = [w for w in line if not w in stop_words]
    return line

#Function to remove Punctuation and normalize characters to lowercase
def remove_punc_lower(line):
    line="".join([char for char in line if char not in string.punctuation])
    line=line.lower()
    return line

# Function to Tokenize words
def tokenize(line):
    tokens = re.split('\W+', line)
    return tokens


if __name__ == "__main__":
    # Combine provided two files
    filenames = ['neg.txt', 'pos.txt']
    with open('../input/file', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


    input_path = '../input/file.txt'
    f=open(input_path)
    text=f.read().split('\n')
    text_with_stop=[]
    for i in range (len(pos)):
        p=tokenize(remove_punc_lower(text[i]))
        text_with_stop.append(p)


    # Creates the relevant phrases from the list of sentences:
    phrases = Phrases(text_with_stop, min_count=30, progress_per=10000)
    # The goal of Phraser() is to cut down memory consumption of Phrases()
    # by discarding model state not strictly needed for the bigram detection task
    bigram = Phraser(phrases)
    #Transform the corpus based on the bigrams detected
    sentences = bigram[text_with_stop]

    # Gensim Word2Vec Implementation:
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer
    w2v_model = Word2Vec(min_count=20,
                         window=2,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores-1)

    # Building the Vocabulary Table:
    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    # Training of the model:
    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    # Exploring the model
    w2v_model.most_similar(positive=['good'], topn=20)
    w2v_model.most_similar(negative=['bad'], topn=20)