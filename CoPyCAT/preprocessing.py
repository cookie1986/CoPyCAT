import json
import pandas as pd

def create_speech_turns(file_path):
    '''
    Receives either a .txt, .csv, or .json file with speaker defined labels.
    
    Returns a 3xN dataframe, where N refers to the number of speech turns. 
    Columns denote (i) 'turn_id' ~ a speech turn ID (a count of speech turns to that point
    in the conversation), (ii) 'speaker' ~ a standardised speaker id, and (iii) the 
    content of the speech turn.
    '''

    if file_path.endswith('.txt'):
        with open(file_path) as file:
            transcript = file.readlines()
            transcript = [line.split(':') for line in  transcript]
    elif file_path.endswith('.csv'):
        # STEP -> SETUP AS DATAFRAME
        print(file_path)
    elif file_path.endswith('.json'):
        # STEP -> SETUP AS DATAFRAME
        print(file_path)
    else:
        raise TypeError("Incorrect file type. Please enter file with either '.txt' or '.csv' extension")
            
    return pd.DataFrame(transcript, columns=['speaker','content'])