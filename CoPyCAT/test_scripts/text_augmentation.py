'''
After prior processing, this script transforms the training region by augmenting copies of each transcript, and applying minor transformations.

Version 2: removes internal functions from inside the main text_augmentation function
'''

import random
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def set_mask(turn, prob):
    '''
    Parameters
    ----------
    turn : list of tokenized words
    p : probability of success in a bernoulli trial

    Returns
    -------
    mask : trial outcomes, 1=success, 0=fail

    '''
    # get maximum number of transformations, based on length of turn.
    max_transform = int(prob*len(turn))

    # create mask of trial outcomes per each word in turn
    mask = np.random.binomial(n=1, p=prob, size=len(turn))
    # alter mask if count of successes exceeds maximum number of transformations
    if np.count_nonzero(mask) > max_transform:
        new_mask = []
        success_counter = 0
        for outcome in mask:
            if outcome == 1:
                success_counter+=1
                if success_counter <= max_transform:
                    new_mask.append(outcome)
                else:
                    outcome = 0
                    new_mask.append(outcome)
            else:
                new_mask.append(outcome)
        mask = np.array(new_mask)  

    return mask

# internal function - bernoulli trial for each word per turn in transcript
def random_deletion(speaker, 
                   turn,  
                   prob = 0.5,
                   tokenType='all',
                   inelig_speakers=[]):
    '''
    turn = tokenized list of POS transformed words (strings)
    method = rd(random deletion)
    prob = float - probability of bernoulli success 
    '''
    
    # check speaker ID and return if speaker 2
    if speaker == 2:
        return turn
    else:
        # get list of stopwords
        stopWords = list(set(stopwords.words('english')))
        
        # calculate token transformations
        transforms = set_mask(turn, prob)
        
        # perform transformation on bernoulli successes
        transformed_turn = []
        for token, success in zip(turn, transforms):
            if tokenType != 'all':
                if token[1] != tokenType:
                    transformed_turn.append(token)
                else:
                    if success == 0:
                        transformed_turn.append(token)
                    elif token[0] in stopWords:
                        transformed_turn.append(token)
            else:
                # if tokenType = A, all POS tags are eligible for deletion
                if success == 0:
                    transformed_turn.append(token)
                elif token[0] in stopWords:
                    transformed_turn.append(token)
                    
        return [speaker, transformed_turn]

# internal function - bernoulli trial for each word per turn in transcript
def synonym_replacement (speaker,
                          turn,
                          prob = 0.5,
                          tokenType='all',
                          inelig_speakers=[]):
        
    # check speaker ID and return if speaker 2
    if speaker == 2:
        return turn
    else:
        # get list of stopwords
        stopWords = list(set(stopwords.words('english')))

        # calculate token transformations
        transforms = set_mask(turn, prob)

        # perform transformation on bernoulli successes
        transformed_turn = []
        for lem, success in zip(turn, transforms):
            if tokenType != 'all':
                if lem[1] != tokenType:
                    transformed_turn.append(lem)
                else:
                    if success == 0:
                        transformed_turn.append(lem)
                    elif lem[0] in stopWords:
                        transformed_turn.append(lem)
                    else:
                        # get list of synonyms per token
                        synList = [s.lemma_names() for s in wordnet.synsets(lem[0])]
                        # flatten syns into single list
                        synListFlat = list(set([syn.lower() for sub in synList for syn in sub]))
                        # remove original term from synonym list
                        synListFlat = [syn for syn in synListFlat if syn != lem[0]]
                        # remove any multi-string tokens
                        synListFlat = [syn for syn in synListFlat if '_' not in syn]
                        # if no synonyms are available, return the original token
                        if len(synListFlat) <1:
                            transformed_turn.append(lem)
                        else:
                            synReplace = random.choice(synListFlat)
                            transformed_turn.append((synReplace, lem[1]))
            else:
              # if tokenType = A, all POS tags are eligible for synonym replacement
                if success == 0:
                    transformed_turn.append(lem)
                elif lem[0] in stopWords:
                        transformed_turn.append(lem)
                else:
                    # get list of synonyms per token
                    synList = [s.lemma_names() for s in wordnet.synsets(lem[0])]
                    # flatten syns into single list
                    synListFlat = list(set([syn.lower() for sub in synList for syn in sub]))
                    # remove original term from synonym list
                    synListFlat = [syn for syn in synListFlat if syn != lem[0]]
                    # remove any multi-string tokens
                    synListFlat = [syn for syn in synListFlat if '_' not in syn]
                    # if no synonyms are available, return the original token
                    if len(synListFlat) <1:
                        transformed_turn.append(lem)
                    else:
                        synReplace = random.choice(synListFlat)
                        transformed_turn.append((synReplace, lem[1]))  
        return [speaker, transformed_turn]
    
# internal function - bernoulli trial for each word per turn in transcript
def synonym_insertion (speaker,
                          turn,
                          prob = 0.5,
                          tokenType='all',
                          inelig_speakers=[]):
    
    def random_insert(lst, item):
        '''
        Appends a list with an element at a random location
        
        Parameters
        ----------
        lst : any list
        item : any object

        Returns
        -------
        lst : appended list
        '''
        
        lst.insert(random.randrange(len(lst)+1), item)

        return lst
    
    # check speaker ID and return if speaker 2
    if speaker == 2:
        return turn
    else:
        # get list of stopwords
        stopWords = list(set(stopwords.words('english')))

        # calculate token transformations
        transforms = set_mask(turn, prob)

        # perform transformation on bernoulli successes
        transformed_turn = []
        for lem, success in zip(turn, transforms):
            if tokenType != 'all':
                if lem[1] != tokenType:
                    transformed_turn.append(lem)
                else:
                    if success == 0:
                        transformed_turn.append(lem)
                    elif lem[0] in stopWords:
                        transformed_turn.append(lem)
                    else:
                        # get list of synonyms per token
                        synList = [s.lemma_names() for s in wordnet.synsets(lem[0])]
                        # flatten syns into single list
                        synListFlat = list(set([syn.lower() for sub in synList for syn in sub]))
                        # remove original term from synonym list
                        synListFlat = [syn for syn in synListFlat if syn != lem[0]]
                        # remove any multi-string tokens
                        synListFlat = [syn for syn in synListFlat if '_' not in syn]
                        # if no synonyms are available, return the original token
                        if len(synListFlat) <1:
                            transformed_turn.append(lem)
                        else:
                            # return the original word
                            transformed_turn.append((lem[0], lem(1)))
                            # select synonym at random
                            synReplace = random.choice(synListFlat)
                            # insert synonym into random point in the turn
                            transformed_turn = random_insert(transformed_turn, (synReplace, lem[1]))
            else:
              # if tokenType = all, all POS tags are eligible for synonym insertion
                if success == 0:
                    transformed_turn.append(lem)
                elif lem[0] in stopWords:
                        transformed_turn.append(lem)
                else:
                    # get list of synonyms per token
                    synList = [s.lemma_names() for s in wordnet.synsets(lem[0])]
                    # flatten syns into single list
                    synListFlat = list(set([syn.lower() for sub in synList for syn in sub]))
                    # remove original term from synonym list
                    synListFlat = [syn for syn in synListFlat if syn != lem[0]]
                    # remove any multi-string tokens
                    synListFlat = [syn for syn in synListFlat if '_' not in syn]
                    # if no synonyms are available, return the original token
                    if len(synListFlat) <1:
                        transformed_turn.append(lem)
                    else:
                        # return the original word
                        transformed_turn.append(lem)
                        # select synonym at random
                        synReplace = random.choice(synListFlat)
                        # insert synonym into random point in the turn
                        transformed_turn = random_insert(transformed_turn, (synReplace, lem[1]))
        return [speaker, transformed_turn]

# test section
# test = ['f',[('dogs','n'),('and','n'),('cats','n'),('are','v'),('big','a'),('animals','n')]]
# # test2 = ['f',[('dog','n'),('and','n')]]

# print(synonym_insertion(speaker=test[0], turn=test[1], prob=0.5))