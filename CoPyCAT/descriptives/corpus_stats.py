import os
import glob
import pandas as pd
import regex as re
from nltk.tokenize import word_tokenize

# main DF to store demographics
demographics = pd.DataFrame(columns=['filename',
                                     'turns',
                                     'ir_turns',
                                     'ie_turns',
                                     'words',
                                     'ir_words',
                                     'ie_words'])
# load each interview
for file in glob.glob('MT*.csv'):
    filename = file[-9:-4:]
    
    # read csv file
    tr = pd.read_csv(file, encoding='utf-8', usecols=['speaker','content'])
    
    # remove punctuation and convert to lowercase
    tr['content'] = tr['content'].apply(lambda x: re.sub(r"[^\w\d\s]","", x.lower()))

    # tokenize
    tr['content'] = tr['content'].apply(lambda x: word_tokenize(x))
    
    # get turn length
    tr['length'] = tr['content'].apply(lambda x: len(x))
    
    # create speaker masks
    ir = tr['speaker']==1
    ie = tr['speaker']==2
    
    # add to demographics DF
    demographics=demographics.append({'filename':filename,
                         'turns': len(tr),
                         'ir_turns':len(tr[ir]),
                         'ie_turns':len(tr[ie]),
                         'words': tr['length'].sum(),
                         'ir_words':tr[ir]['length'].sum(),
                         'ie_words':tr[ie]['length'].sum()}, ignore_index=True)
    