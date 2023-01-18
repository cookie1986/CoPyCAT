import re
import json
# import spacy
import pandas as pd

# nlp = spacy.load("en_core_web_sm")

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

    # STEP 1 - STANDARDISE SPEAKER LABELS

    # STEP 2 - MERGE ADJACENT SPEECH TURNS

    # STEP 3 - REMOVE NON-ASCIII CHARS

    # STEP 4 - REMOVE LIST OF COMMON ANNOTATIONS (I.E., LAUGHTER)
    ## excluded_terms is a user supplied list of terms to exclude, if None use common annotaions

    # STEP 5 - REMOVE UMS AND AHS

    # STEP 6 - STANDARDISE POSSESSIVE NOUNS

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