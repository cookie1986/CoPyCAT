import os
import glob
import json
import pandas as pd
import numpy as np
import regex as re
import spacy
from unidecode import unidecode

from .contractions import word_contractions

nlp = spacy.load("en_core_web_sm")

def expand_tokens(text, compiler, tokenDict):
    def replace(match):
        return tokenDict[match.group(0)]
    return compiler.sub(replace, text)

def create_dataframe(file_path):
    '''
    Receives either a .txt, .csv, or .json file with speaker defined labels.
    
    Returns a 3xN dataframe, where N refers to the number of speech turns. 
    Columns denote (i) 'turn_id' ~ a speech turn ID (a count of speech turns to that point
    in the conversation), (ii) 'speaker' ~ a standardised speaker id, and (iii) the 
    content of the speech turn.
    '''

    if file_path.endswith('.txt'):
        print(file_path)
        with open(file_path) as file:
            transcript = file.readlines()
            transcript = [line.split(':') for line in  transcript]
            transcript = pd.DataFrame(transcript, columns=['speaker','content'])
            transcript['turn_id'] = range(len(transcript))    

        return transcript

    elif file_path.endswith('.csv'):
        print(file_path)
        transcript = pd.read_csv(file_path)   

        return transcript  
    
    elif file_path.endswith('.json'):
        print(file_path)
        transcript = pd.read_json(file_path)

        return transcript

    else:
        raise TypeError("Incorrect file type. Please enter file with either '.txt' or '.csv' extension")    
        


def basic_cleaning(df, excluded_terms = None):
    '''
    Receives a dataframe of speech turns with non-standardised speaker labels.
    
    Standardises speaker labels so that the first speaker = 1, and the second
    speaker = 2. Merges adjacent speech turns produced by the same speaker.
    Performs basic cleaning of spoken content.

    Returns standardised dataframe.
    '''

    # set unicode characters as ASCII
    transcript = unidecode(transcript)
    
    # standardize key formatting
    transcript = re.sub(r"\[", "(", transcript) # open square brackets to open paranthesis
    transcript = re.sub(r"\]", ")", transcript) # closed square brackets to closed paranthesis
    transcript = re.sub(r"[!?]", ".", transcript) # standardize sentence boundaries 
    transcript = re.sub(r"\s?(\.){3}", ".", transcript) # hesitation marker
    transcript = re.sub(r"`", "'", transcript) # standardize sentence boundaries
    transcript = re.sub(r"^SEN. |^REP. |^SEC. |^GOV. ", "", transcript, flags=re.M|re.I)
    
    # transform common forward slash uses in corpus
    transcript = re.sub(r"9/11", "September 11 attacks", transcript)
    transcript = re.sub(r"1/2", "one-half", transcript)
    transcript = re.sub(r"1/3", "one-third", transcript)
    transcript = re.sub(r"1/10(th)?", "one-tenth", transcript)
    transcript = re.sub(r"1/50(th)?", "one-fiftieth", transcript)
    transcript = re.sub(r"50/50", "fifty-fifty", transcript)
    transcript = re.sub(r"20/20", "twenty-twenty", transcript)
    transcript = re.sub(r"24/7", "twenty-four-seven", transcript)
    transcript = re.sub(r"shia/sunni", "shia-sunni", transcript, flags=re.I)
    transcript = re.sub(r"\/", " ", transcript, flags=re.I)
    
    # remove timestamps
    transcript = re.sub(r"\(\d{1,2}:\d{2}(:\d{2})?\)", "", transcript)
    
    # remove single-line annotations
    transcript = re.sub(r"\(crosstalk(\s[^\(]*)?\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(overtalk\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(laugh[^\(]*\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\((inaudible|indistinct|unintelligible)[^\(]*\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\((ph|sic|sigh|coughing|k)?\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(cheer[^\(]*\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(applause\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(boo[^\(]*\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\((d[-\w]*|r[-\w]*)\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(audio (cut|gap){1}\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(expletive[^\(]*\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(affirmative\)", "", transcript, flags=re.I)
    transcript = re.sub(r"\(commercial break\)", "", transcript, flags=re.I)
    
    # standardize pre-recorded segment markers
    transcript = re.sub(r"\((begin[^\(]*)\)", "(BEGIN_SEG)", transcript, flags=re.I)
    transcript = re.sub(r"\((end[^\(]*)\)", "(END_SEG)", transcript, flags=re.I)
    # remove pre-recorded segments
    transcript = re.sub(r"\(BEGIN_SEG\)[^\(]*\(END_SEG\)", "", transcript, flags=re.I)
    
    # remove remaining use of brackets
    transcript = re.sub(r"[\(\)]", "", transcript)
    
    # identify false starts
    transcript = re.sub(r'(?<=[a-z])\s?-{1,3}(?=\s[a-z])', ' (FS)', transcript, flags=re.I)
    
    # expand contracted phrases
    transcript = expand_tokens(transcript, compiler = word_contractions, tokenDict = js_dict)

    # isolate speaker ID formatting
    transcript = re.sub(r"(?<=[A-Z\s]):", "___", transcript)

    # split raw string into nested list of speaker/speech per turn in transcript
    splitter = r'^((?:[^\W\d_]|[^\S\r\n])+)___(.*(?:\n(?!(?:[^\W\d_]|[^\S\r\n])+___).*)*)'
    transcript = [[x.strip(),y.replace('\n',' ').strip()] for x,y in re.findall(splitter, transcript, re.M)]
    
    # set nested list to dataframe
    transcript = pd.DataFrame(transcript, columns = ['speaker','content'])
    
    # check for empty rows (and drop if found)
    transcript['len'] = transcript['content'].apply(lambda x: len(x))
    transcript = transcript[transcript['len'] != 0]
    transcript = transcript.drop('len', axis=1)

    # standardize speaker labels
    firstSpeaker = transcript['speaker'].iat[0]
    transcript['speaker'] = np.where(transcript['speaker'].eq(firstSpeaker), 1, 2)
    
    # merge adjacent messages sharing the same speaker ID
    # compare each speaker with the previous
    transcript['prev_speaker'] = transcript['speaker'].shift(1).mask(pd.isnull, transcript['speaker'])
    transcript['speaker_change'] = np.where(transcript['speaker'] != transcript['prev_speaker'], 'True','False')
    # create list to store speaker change keys
    speaker_changes = []    
    # start speaker change counter
    speaker_change_counter =1
    # loop through each message to record speaker changes
    for i in transcript['speaker_change']:
        if i == 'False':
            speaker_changes.append(speaker_change_counter)
        else:
            speaker_change_counter+=1
            speaker_changes.append(speaker_change_counter)
    transcript['chunking'] = speaker_changes
    # join adjacent messages sharing the same speaker ID 
    transcript['content'] = transcript.groupby(['chunking'])['content'].transform(lambda x: ' '.join(x))
    # drop duplicate rows based on message content and conversation chunk
    transcript = transcript.drop_duplicates(subset=['content','chunking'])
    # drop non-needed columns
    transcript = transcript[['speaker','content']]

    return df

def nlp_preproc(speech_turn):
    '''
    Receives a string representing a speech turn.

    Returns tokenized, POS tagged, and lemmatized speech
    turn as str.
    '''

    # STEP 1 - TOKENIZE
    # STEP 2 - POS TAG
    # STEP 3 - LEMMATIZE

    return speech_turn