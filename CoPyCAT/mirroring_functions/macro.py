### STYLISTIC MACRO
import os
import re
import numpy as np
import pandas as pd


# read in rLSM raw data
data = pd.read_csv('rlsm_rawdata.csv')

# get set of filenames
filename_list = list(set(data['Filename']))

# empty list to store interviews
corpus_list = []

# empty DF for rlsm scores 
ir_rlsm = pd.DataFrame()

# split file into constitutent interviews
for i in filename_list:
    mask = data['Filename'] == i
    tr = data[mask].reset_index(drop=True)
    # change speaker IDs
    tr['Speaker'] = tr['Speaker'].apply(lambda x: 'IR' if x == 1 else 'IE')
    # add to main list
    corpus_list.append([i, tr])
    
# loop through each DF in list
for df in corpus_list:
    filename = df[0]
    tr = df[1]
    
    # compare with previous
    cols = list(tr.iloc[:, 3:])
    for i in cols:
        tr[i+'_prev'] = tr[i].shift().fillna('').to_numpy()
    # drop IE rows
    tr = tr[tr['Speaker']=='IR'][1:].reset_index(drop=True)
    
    ir_rlsm=ir_rlsm.append(tr)

# export rlsm for the IR to calculate in excel
ir_rlsm.to_csv('rlsm_prep.csv')


### SYNTACTIC MACRO
import os
import glob
import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize, sent_tokenize, RegexpParser 

acc_vals = pd.DataFrame()
   

def parse_trees(string):

    # Extract all parts of speech from any text 
    chunker = RegexpParser(""" 
                           NP: {<DT>?<JJ>*<NN>}    #To extract Noun Phrases 
                           P: {<IN>}               #To extract Prepositions 
                           V: {<V.*>}              #To extract Verbs 
                           PP: {<P> <NP>}          #To extract Prepostional Phrases 
                           VP: {<V> <NP|PP>*}      #To extarct Verb Phrases 
                           """) 
    
    # create empty array to store mean branching factor
    mean_bf = []
    
    # sentence tokenizer
    sents = sent_tokenize(string)
    
    # loop through each sentence
    for s in sents:
        # Find all parts of speech in string
        tagged = pos_tag(word_tokenize(s))
    
        # parse sentence
        parsed_string = chunker.parse(tagged)
    
        # get list of subtrees
        subtree_list = []
        for subtree in parsed_string.subtrees():
            subtree_list.append(subtree)
    
        # add sentence-level bf to turn-level
        mean_bf.append(np.mean([len(i) for i in subtree_list]))
        
    # get turn level mean
    mean_bf = np.mean(mean_bf)
    
    return mean_bf


# iterate through each interview
for file in glob.glob('MT*.csv'):
    filename = file[-9:-4]
    tr = pd.read_csv(file, encoding='utf-8')
    
    # convert turn to parse trees and get average branching factor
    tr['mean_bf'] = tr['content'].apply(lambda x: parse_trees(x))
    
    # compare turn with previous
    tr['prev'] = tr['mean_bf'].shift().fillna('')
    
    # mask interviewer turns
    ir = tr[tr['speaker']==1][1:]
    
    # get difference between turn and prev
    ir['bf_diff'] = abs(ir['mean_bf'].subtract(ir['prev']))
    
    acc_vals=acc_vals.append({'filename': filename,
                     'mean_bf_diff': np.mean(ir['bf_diff']),
                     'min_bf_diff': np.min(ir['bf_diff']),
                     'max_bf_diff': np.max(ir['bf_diff'])}, ignore_index=True)
    

    ### SEMANTIC MACRO
    import os
import re
import glob
import stanza
import warnings
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

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

# remove stopwords
stop_words = list(stopwords.words('english'))

# set directory
os.chdir('C:/ProgramFiles/glove')
# create dictionary to store word embeddings
embeddings_dict = {}

# open word embeddings file and set as dictionary
with open("glove.6B.50d.txt", 'r', encoding="utf-8-sig") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
    f.close()
    
# internal function to get word embeddings
def word_to_msg_vec(word_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # empty list to store word embeddings
        word_emd_list = []
        # get vectors for each word and store
        for word in word_list:
            vec = embeddings_dict.get(word)
            if vec is not None:
                word_emd_list.append(vec) 
        # average word embeddings in message to get msg-level vector
        msg_vec = np.average(word_emd_list, axis=0)
    
    return msg_vec

features = pd.DataFrame(columns=['filename','cosine_mean','cosine_max','cosine_min'])

# read in dataframes
for f in glob.glob("MT*.csv"):
    tr = pd.read_csv(f, encoding='utf-8')
    # set filename
    filename = f[:-4]
    
    # remove any remaining punctuation and convert to lowercase
    tr['content'] = tr['content'].apply(lambda x: re.sub(r"[^\w\s\d]","", x.lower()))
    # tokenize, pos tag and lemmatize
    tr['content'] = tr['content'].apply(lambda x: list(get_lemma(x)) if len(x) > 1 else x)
    # remove false start label
    tr['content'] = tr['content'].apply(lambda x: [token for token in x if token != "fs"])
    # remove stopwords
    tr['content'] = tr['content'].apply(lambda x: [w for w in x if w not in stop_words])
    
    # convert turn to vector
    tr['content'] = tr['content'].apply(lambda x: word_to_msg_vec(x))
    # set any NaNs to zero
    for row in tr.loc[tr.content.isnull(), 'content'].index:
        tr.at[row, 'content'] = np.zeros(50)
    # calculate cosine similarity between each turn n and n-1
    tr['content'] = [cosine_similarity(tr['content'].iloc[i].reshape(1,-1), tr['content'].iloc[i-1].reshape(1,-1)) for i in tr.index]
    
    # mask over IR turns only
    ir = tr['content'][tr['speaker']==1].iloc[1:]
    ir = ir.loc[ir>0]
    
    # add values to features dataframe
    sim_scores = {'filename':filename,
                  'cosine_mean':float(np.mean(ir[1:])),
                  'cosine_max':float(np.max(ir[1:])),
                  'cosine_min':float(np.min(ir[1:]))}
    features=features.append(sim_scores, ignore_index=True)

    # print status update
    print("file: "+filename+" is done")