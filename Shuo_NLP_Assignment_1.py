import pandas as pd
import re
import codecs
import sys
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string

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
    input_path = 'pos.txt'
    f=open(input_path)
    pos=f.read().split('\n')
    pos_stop=[]
    pos_no_stop=[]
    for i in range (len(pos)):
        p=tokenize(remove_punc_lower(pos[i]))
        pos_no_stop.append(p)
        p=remove_stopwords(p)
        pos_stop.append(p)
    train_list,val_list=train_test_split(pos_stop,train_size=0.8,random_state=0)
    val_list,test_list=train_test_split(val_list,test_size=0.5,random_state=0)
    np.savetxt("train.csv", train_list, delimiter=",",fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')
    train_list_no_stopword,val_list_no_stopword=train_test_split(pos_no_stop,train_size=0.8,random_state=0)
    val_list_no_stopword,test_list_no_stopword=train_test_split(val_list_no_stopword,test_size=0.5,random_state=0)
    np.savetxt("train_no_stopword.csv", train_list_no_stopword,delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopword,delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopword,delimiter=",", fmt='%s')
