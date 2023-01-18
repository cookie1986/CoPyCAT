import os
import re
import glob
import json
import spacy
import pandas as pd

# load SpaCy language model
nlp = spacy.load("en_core_web_lg")

# load interviewer names dictionary
with open('') as t: 
    namesFile = t.read() 
    names_dict = json.loads(namesFile)

# internal function to generate named entities from string
def getNER(speech_content):
    nlp_obj = nlp(speech_content)
    cats = ['PERSON','NORP','FAC','ORG','GPE','LOC','EVENT','LAW','DATE','TIME','MONETARY','QUANTITY']
    namedEnts = [((token.text),(token.label_)) for token in nlp_obj.ents if token.label_ in cats]
    
    return namedEnts

# create empty list to store scores
outcome = []

# load each interview
for file in glob.glob('MT*.csv'):
    filename = file[-9:-4:]
    interview_ref = filename

    # read csv file
    tr = pd.read_csv(file, encoding='utf-8', usecols=['speaker','content'])
    
    # return name of the interviewer
    ir = names_dict[interview_ref]
    
    # filter on turns generated by politicians
    pol_turns = tr[tr['speaker'] == 2]
    
    # filter pol turns and remove any references to the interviewer
    pol_turns['content'] = pol_turns['content'].apply(lambda x: x.replace(ir, ''))
    
    # get summation of interviewee responses
    pol_turns_global = '.'.join(x for x in pol_turns['content'].tolist())
    
    # remove false starts
    pol_turns_global = re.split(r"\s", pol_turns_global)
    for i, word in enumerate(pol_turns_global):
        if word == "(FS)":
            if pol_turns_global[i-1] == pol_turns_global[i+1]:
                pol_turns_global[i+1] = ""
            pol_turns_global[i] = ""
    pol_turns_global = " ".join(x for x in pol_turns_global if x)
    # remove other punctuation
    pol_turns_global = re.sub(r"-{2,3}", "", pol_turns_global)
    
    # get noun phrases
    noun_phrases = []
    doc = nlp(pol_turns_global)
    for chunk in doc.noun_chunks:
        noun_phrases.append(chunk.text.lower())
    noun_phrases = list(set(noun_phrases))
    
    # get a list of named entities spoken by the politician
    named_ents = list(set(getNER(pol_turns_global)))
    
    # get global specificity score
    specificity = len(named_ents)/len(noun_phrases)*100
    
    # take count of entity set and store as a variable
    outcome.append([filename, round(specificity,2)])
    
# load scores as a dataframe 
outcomeDF = pd.DataFrame(outcome, columns=['filename','score'])