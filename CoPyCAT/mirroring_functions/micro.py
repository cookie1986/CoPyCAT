import os
import re
import glob
import stanza
import numpy as np
import pandas as pd
from nltk import ngrams
from collections import Counter


###  MICRO LEXICAL

# set size of ngrams
ngram_size = 1

# download English model for Stanza
stanza.download('en') 
# initialize English neural pipeline
nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma')
    
# internal function to perform Stanza lemmatization
def get_lemma(string):
    # load string as nlp object
    doc = nlp(string)
    # lemmatize
    lemma = [word.lemma for sent in doc.sentences for word in sent.words]
    
    return lemma

# internal function: get ngrams from token list
def get_ngrams(token_list, num):
    ngram_list = list(ngrams(token_list, num))
    
    ngram_list = ['-'.join(i for i in x) for x in ngram_list]
    
    return ngram_list

# set working dir
os.chdir('C:/Users/Darren Cook/Documents/PhD Research/political_interviews_data/manual_transcripts/basic_cleaning/')

acc_vals = {}

# iterate through each interview
for file in glob.glob('MT*.csv'):
    filename = file[-9:-4]
    tr = pd.read_csv(file, encoding='utf-8')
    
    # drop the first (interviewer) row
    tr = tr[1:]
    # remove any remaining punctuation and convert to lowercase
    tr['content'] = tr['content'].apply(lambda x: re.sub(r"[^\w\s\d]","", x.lower()))
    # tokenize, pos tag and lemmatize
    tr['content'] = tr['content'].apply(lambda x: list(get_lemma(x)) if len(x) > 1 else x)
    # remove false start label
    tr['content'] = tr['content'].apply(lambda x: [token for token in x if token != "fs"])
    # set bigram or trigrams here
    tr['content'] = tr['content'].apply(lambda x: get_ngrams(x, ngram_size))
    # take set of each turn
    tr['content'] = tr['content'].apply(lambda x: list(set(x)))    
    
    # compare each turn with the previous
    tr['previous'] = tr['content'].shift().fillna('')
    # get the intersection
    tr['shared'] = tr.apply(lambda x: np.intersect1d(x.content, x.previous), axis=1)

    # calculate subtrahend
    # get the token set of IR speech
    ir = tr[tr['speaker'] ==1]
    # collate all n-grams into one list
    ir_turn_set = [token for turn in ir['content'] for token in turn]
    # create counter object of each n-gram
    subtrahend = Counter(token for token in ir_turn_set)
    # divide each value by the number of IR turns
    for token, counter in subtrahend.items():
        subtrahend[token] /= len(ir)
        
    # calculate minuend
    # collate shared n-grams into one list
    ir_shared_set = [token for turn in ir['shared'] for token in turn]
    # create counter object of each n-gram
    shared_counter = Counter(token for token in ir_shared_set)
    # get denominator
    ie_turn_set = [token for turn in ir['previous'] for token in turn]
    ie_count = Counter(token for token in ie_turn_set)
    # get final value for minuend
    minuend = {k : v / ie_count[k] for k, v in shared_counter.items() if k in ie_count}  
    # identify non-converged tokens
    tokenDiff = list(set(ir_turn_set).difference(ir_shared_set))
    # create dictionary where the conditional probability is zero
    divergedTokens = dict.fromkeys(tokenDiff, float(0))
    minuend.update(divergedTokens)
    
    # subtract subtrahend from minuend to get final ngram score
    acc_scores = {k : v - subtrahend[k] for k, v in minuend.items() if k in subtrahend}

    # add values to main list
    acc_vals.update({filename: acc_scores})
    
    # print status update
    print("file: "+filename+" is done")

# set as feature vector
feature_vec = pd.DataFrame.from_dict(acc_vals, orient='index').fillna(0)

# remove features appearing in less than 10% of corpus
min_nonZero = int(len(feature_vec)*0.1)
feature_vec_reduced = feature_vec.loc[:, (feature_vec.replace(0, np.nan).notnull().sum(axis=0) >= min_nonZero)]

# sum over absolute column values to find the top X features
x = 200
abs_vals = pd.Series(feature_vec_reduced.abs().sum().sort_values(ascending=False))
abs_vals = abs_vals[:x]
col_index = list(abs_vals.index)
feature_vec_reduced = feature_vec_reduced[col_index]



### SYNTACTIC MICRO
import os
import re
import glob
import spacy
import stanza
import numpy as np
import pandas as pd
from collections import Counter

# download English model for Stanza
stanza.download('en') 
# initialize English neural pipeline
nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma')

# load spacy library
spacy_nlp = spacy.load('en_core_web_sm')

# internal function: get list of dependency relation tags
def get_depparse_tags(string):
    # create spacy object based on turn
    turn = spacy_nlp(string)
    # get turn set of relation tags
    tag_list = list(set([token.dep_ for token in turn if token.dep_ not in ['ROOT', 'punct']]))
    
    return tag_list

# useful for checking the meaning of each dep tag
spacy.explain('nsubj')


acc_vals = {}

# iterate through each interview
for file in glob.glob('MT*.csv'):
    filename = file[-9:-4]
    tr = pd.read_csv(file, encoding='utf-8')
    
    # drop first (interviewer) row
    tr = tr[1:]    
    # remove false starts
    tr['content'] = tr['content'].apply(lambda x: re.sub(r"\(FS\)","", x))
    
    # dependency parse
    tr['content'] = tr['content'].apply(lambda x: get_depparse_tags(x))
    
    # compare each turn with the previous
    tr['previous'] = tr['content'].shift().fillna('')
    # get the intersection
    tr['shared'] = tr.apply(lambda x: np.intersect1d(x.content, x.previous), axis=1)
    
    # calculate subtrahend
    # get the token set of IR speech
    ir = tr[tr['speaker'] ==1]
    # collate all n-grams into one list
    ir_turn_set = [token for turn in ir['content'] for token in turn]
    # create counter object of each n-gram
    subtrahend = Counter(token for token in ir_turn_set)
    # divide each value by the number of IR turns
    for token, counter in subtrahend.items():
        subtrahend[token] /= len(ir)
        
    # calculate minuend
    # collate shared n-grams into one list
    ir_shared_set = [token for turn in ir['shared'] for token in turn]
    # create counter object of each n-gram
    shared_counter = Counter(token for token in ir_shared_set)
    # get denominator
    ie_turn_set = [token for turn in ir['previous'] for token in turn]
    ie_count = Counter(token for token in ie_turn_set)
    # get final value for minuend
    minuend = {k : v / ie_count[k] for k, v in shared_counter.items() if k in ie_count}  
    # identify non-converged tokens
    tokenDiff = list(set(ir_turn_set).difference(ir_shared_set))
    # create dictionary where the conditional probability is zero
    divergedTokens = dict.fromkeys(tokenDiff, float(0))
    minuend.update(divergedTokens)
    
    # subtract subtrahend from minuend to get final ngram score
    acc_scores = {k : v - subtrahend[k] for k, v in minuend.items() if k in subtrahend}

    # add values to main list
    acc_vals.update({filename: acc_scores})
    
    # print status update
    print("file: "+filename+" is done")

# set as feature vector
feature_vec = pd.DataFrame.from_dict(acc_vals, orient='index').fillna(0)




### STYLISTIC MICRO
## prep

import os
import glob
import pandas as pd
import regex as re

# tempt store for DFs
corpus_list = []

# load each interview
for file in glob.glob('MT*.csv'):
    filename = file[-9:-4:]
    
    # read csv file
    tr = pd.read_csv(file, encoding='utf-8', usecols=['speaker','content'])
    
    # remove first turn
    tr = tr[1:]
    
    # remove false start annotation
    tr['content'] = tr['content'].apply(lambda x: re.sub(r"\(FS\)","", x))
    
    # remove punctuation and convert text to lowercase
    tr['content'] = tr['content'].apply(lambda x: re.sub(r"[^\w\s\d]","", x.lower()))
    
    # add turn counter
    tr['turn'] = range(len(tr))
    
    # add filename column
    tr['filename'] = filename

    # reorder columns
    tr=tr[['filename','turn','speaker','content']]

    # add each corpus to main list
    corpus_list.append(tr)
    
# concatenate into one main DF
liwc_prep = pd.concat(i for i in corpus_list)


## calculation
import os
import numpy as np
import pandas as pd

# read in liwc results file
liwc_res = pd.read_csv('liwc_output.csv')

# get set of filenames
filename_list = list(set(liwc_res['A']))

# empty list to store interviews
corpus_list = []

acc_vals = {}

# split file into constitutent interviews
for i in filename_list:
    mask = liwc_res['A'] == i
    tr = liwc_res[mask].reset_index(drop=True)
    # drop non-essential columns
    tr=tr.drop(['A','B','D'], axis=1)
    # rename first two columns
    tr.rename(columns = {'C':'speaker'}, inplace = True)
    # change speaker IDs
    tr['speaker'] = tr['speaker'].apply(lambda x: 'IR' if x == 1 else 'IE')
    # remove non-needed columns
    tr = tr.drop(tr.columns[[1,2,3,4,5,6,7,8]], axis = 1)
    # binarize feature columns
    tr.iloc[:,1:] = np.heaviside(tr.iloc[:,1:],0).astype(int)
    # add to main list
    corpus_list.append([i, tr])
    

for df in corpus_list:
    filename = df[0]
    tr = df[1]
    # create speaker subsets
    ir = tr['speaker'] == 'IR'
    ie = tr['speaker'] == 'IE'
    irDF = tr[ir].drop('speaker', axis=1).reset_index(drop=True)
    ieDF = tr[ie].drop('speaker', axis=1).reset_index(drop=True)
    ieDF = ieDF.iloc[:len(irDF)] # remove any rows after the last IR row
    
    # get column names
    columns = list(irDF.columns[:]) # create list of columns
    
    # calculate minuend
    # empty dict for ie column values
    ie_dict = {}
    
    # get denominator
    for i in columns:
        ie_dict.update({i: ieDF[i].sum()})    
    
    # combine speaker DFs
    combineDF = irDF+ieDF
    # empty dict for join column values
    conv_dict = {}
    # count occurences of convergence - numerator
    for i in columns:
        combineDF2_bool = combineDF[i]==2
        combineDF2 = combineDF[i][combineDF2_bool]
        conv_dict.update({i: len(combineDF2)})  
    
    # multiply dicts to get conv scores and replace nans
    minuend = {k : v / ie_dict[k] if ie_dict[k] else np.float64(0) for k, v in conv_dict.items() if k in ie_dict}

    # calculate subtrahend
    subtrahend = {}
    for i in columns:
        subtrahend.update({i: irDF[i].sum()/len(irDF)})
        
    # calculate accommodation
    acc_scores = {k : v - subtrahend[k] for k, v in minuend.items() if k in subtrahend}
    
    # add scores to main list
    acc_vals.update({filename:acc_scores})
    
# set as feature vector
feature_vec = pd.DataFrame.from_dict(acc_vals, orient='index').fillna(0)