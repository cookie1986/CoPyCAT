import os
import re
import glob
import stanza
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


# create dictionary to store word embeddings
embeddings_dict = {}
# download English model for Stanza
stanza.download('en') 
# initialize English neural pipeline
nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma')

# open word embeddings file and set as dictionary
with open("glove.6B.50d.txt", 'r', encoding="utf-8-sig") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
    f.close()
    
# internal function to perform Stanza lemmatization
def get_lemma(string):
    # load string as nlp object
    doc = nlp(string)
    # lemmatize
    lemma = [word.lemma for sent in doc.sentences for word in sent.words]
    return lemma

# internal function to get word embeddings
def word_to_msg_vec(word_list):
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

# set stopwords list
stop_words = list(stopwords.words('english'))


# create empty list to store scores
outcome = []

# load each interview
for file in glob.glob('MT*.csv'):
    filename = file[-9:-4:]
    interview_ref = filename
    
    # read csv file
    tr = pd.read_csv(file, encoding='utf-8', usecols=['speaker','content'])
    
    # remove punctuation and convert to lowercase
    tr['content'] = tr['content'].apply(lambda x: re.sub(r"[^\w\d\s]","", x.lower()))
    
    # split DF into speaker subsets
    interviewer = tr[tr['speaker']==1]
    interviewee = tr[tr['speaker']==2]
    
    # check length of each turn and merge turns shorter than predefined length
    min_turn = 5
    
    # merge short interviewer rows
    interviewer['length'] = interviewer['content'].apply(lambda x: len(re.split(r"\s",x)))
    interviewer['short_turn'] = np.where(interviewer['length'] < min_turn, 'True','False')
    # create list to store merges based on short turns
    turn_merges = []    
    # start counter
    turn_merge_counter =0
    # loop through each message to record short turns
    for i in interviewer['short_turn']:
        if i == 'True':
            turn_merges.append(turn_merge_counter)
        else:
            turn_merge_counter+=1
            turn_merges.append(turn_merge_counter)
    interviewer['chunking'] = turn_merges
    # join short messages
    interviewer['content'] = interviewer.groupby(['chunking'])['content'].transform(lambda x: ' '.join(x))
    # drop duplicate rows based on message content and conversation chunk
    interviewer = interviewer.drop_duplicates(subset=['content','chunking'])
    # drop non-needed columns
    interviewer = interviewer[['speaker','content']] 
    
    # merge short interviewee rows
    interviewee['length'] = interviewee['content'].apply(lambda x: len(re.split(r"\s",x)))
    interviewee['short_turn'] = np.where(interviewee['length'] < min_turn, 'True','False')
    # create list to store merges based on short turns
    turn_merges = []    
    # start counter
    turn_merge_counter =0
    # loop through each message to record short turns
    for i in interviewee['short_turn']:
        if i == 'True':
            turn_merges.append(turn_merge_counter)
        else:
            turn_merge_counter+=1
            turn_merges.append(turn_merge_counter)
    interviewee['chunking'] = turn_merges
    # join short messages
    interviewee['content'] = interviewee.groupby(['chunking'])['content'].transform(lambda x: ' '.join(x))
    # drop duplicate rows based on message content and conversation chunk
    interviewee = interviewee.drop_duplicates(subset=['content','chunking'])
    # drop non-needed columns
    interviewee = interviewee[['speaker','content']] 
    
    # recombine sub DFs
    tr = pd.concat([interviewer,interviewee]).sort_index()

    # merge adjacent messages sharing the same speaker ID
    # compare each speaker with the previous
    tr['prev_speaker'] = tr['speaker'].shift(1).mask(pd.isnull, tr['speaker'])
    tr['speaker_change'] = np.where(tr['speaker'] != tr['prev_speaker'], 'True','False')
    # create list to store speaker change keys
    speaker_changes = []    
    # start speaker change counter
    speaker_change_counter =1
    # loop through each message to record speaker changes
    for i in tr['speaker_change']:
        if i == 'False':
            speaker_changes.append(speaker_change_counter)
        else:
            speaker_change_counter+=1
            speaker_changes.append(speaker_change_counter)
    tr['chunking'] = speaker_changes
    # join adjacent messages sharing the same speaker ID 
    tr['content'] = tr.groupby(['chunking'])['content'].transform(lambda x: ' '.join(x))
    # drop duplicate rows based on message content and conversation chunk
    tr = tr.drop_duplicates(subset=['content','chunking']).reset_index(drop=True)
    # drop non-needed columns
    tr = tr[['speaker','content']]
    
    # tokenize, pos tag and lemmatize
    tr['content'] = tr['content'].apply(lambda x: list(set(get_lemma(x))))
    
    # remove stopwords
    tr['content'] = tr['content'].apply(lambda x: [w for w in x if w not in stop_words])
    
    # transform turn to message-length vector
    tr['content'] = tr['content'].apply(lambda x: word_to_msg_vec(x))
    
    # set any NaNs to zero
    for row in tr.loc[tr.content.isnull(), 'content'].index:
        tr.at[row, 'content'] = np.zeros(50)
        
    # calculate cosine similarity with the previous row
    tr['cos_sim'] = [cosine_similarity(tr['content'].iloc[i].reshape(1,-1), tr['content'].iloc[i-1].reshape(1,-1)) for i in tr.index]

    # filter on turns generated by politicians
    pol_turns = tr[tr['speaker'] == 2]
    
    # take global relevance score
    relevance = round(float(np.array(pol_turns['cos_sim']).mean()*100),2)
    
    # add to main scores
    outcome.append([filename, relevance])
    
# load scores as a dataframe 
outcomeDF = pd.DataFrame(outcome, columns=['filename','score'])