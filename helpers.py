''' helpers.py

    Contains functions like get_trial_blocks, get_threshold_data and get_cum_score to analyze the
    Mouse_Data generated through Spike2 input.

    @mik-schutte
'''
import numpy as np
import pandas as pd
from queue import Queue
from copy import deepcopy
from Mouse_Data import Mouse_Data

def get_trial_blocks(session_data):
    ''' Creates a list of trials blocks where each block is seperated by a stimulation change
        
        INPUT:
            session_data(pd.DataFrame): dataframe containing data from one individual session e.g. Mouse_Class.all_data[sessionID]   
        OUTPUT:
            blocks(list of pd.DataFrames): a list of session_data blocks cut on intensity change
    '''
    # Get a np.array of all intensities used in the session and find the index of intensity change
    intensity_list = session_data['intensity']
    diff_loc = np.where(np.diff(intensity_list) != 0)[0]
    
    # Now that jump in intensity is located use this to create a list of trial blocks
    blocks = []
    start = 0
    for loc in diff_loc:
        end = loc
        trial_block = session_data.loc[start:end]
        start = end + 1
        blocks.append(trial_block)
    return blocks

def get_threshold(session_data, min_score):
    ''' Gets the lowest intensity that was succesfully detected within the experimental session
    
        INPUT:
            session_data(pd.DataFrame):
            min_score(float): minimal fraction of succesful trials in the trial block
        OUTPUT:
            threshold(int): value of the lowest intensity that was succesfully detected above min_score
    '''
    # Get the trial blocks of that session
    blocks = get_trial_blocks(session_data)
        
    # For each block determine the performance score
    # NOTE this has nothing to do with min_score as this is set in the initiation of Mouse_Class
    threshold_dic = {}
    for block in blocks:
        succes = block.loc[block['succes'] == True]
        score = len(succes)/len(block)
        
        # Check if the score is above the performance requirement and add to dict
        if score >= min_score:
            intensity = block.iloc[0]['intensity']
            threshold_dic[intensity] = block
        
    # From the dict with succesful blocks, get the lowest intensity
    try: #TODO this is not nice coding but prevents crash due to no succesful blocks in session
        threshold = np.min(list(threshold_dic.keys()))  
        return threshold 

    except:
        print(f'Failed at block {block}')

def get_threshold_list(mouse_class, min_score):
    ''' Creats a list of lowest intensity succesful block over sessions
    
        INPUT:
            mouse_class(Mouse_Class): Dataclass with attributes like id, sessions, all_data and concatenated data
            min_score(float): minimal fraction of succesful trials in the trial block
        OUTPUT:
            threshold_list(list): values of the lowest succesfully trial for each session
    '''
    threshold_list = [160]
    for session in mouse_class.sessions:
        session_data = mouse_class.all_data[session]
        threshold = get_threshold(session_data, min_score)
        threshold_list.append(threshold)
        
        # Get if expert already achieved
        if threshold <= 20:
            break
    return threshold_list

def get_threshold_data(mouse_class_list, min_score):
    '''docstring'''
    # Lets get all the threshold lists for the control animals
    threshold_data = []
    for mouse_class in mouse_class_list:
        threshold_list = get_threshold_list(mouse_class, min_score)
        # Fill it up to a certain size for nice plotting 
        counter = len(threshold_list)
        while counter <5:
            threshold_list.append(threshold_list[-1])
            counter = len(threshold_list)
        threshold_data.append(threshold_list)
            
    return threshold_data 

def get_avg_std_threshold(threshold_data, max_sessions=5):
    '''docstring'''
    average_list = []
    std_list = []
    for i in range(max_sessions):
        day_list = [threshold[i] for threshold in threshold_data]
        average = np.mean(day_list)
        average_list.append(average)
        std = np.std(day_list) 
        std_list.append(std)
    return average_list, std_list

def get_cum_score(mouse): #TODO change this to yield all parameters over all sessions
    ''' Calculates the cumulative or learning score progressing over all trials
        
        INPUT:
            mouse(Mouse_Data): class with attributes like id, sessions, all_data and concatenated data
        OUPUT:
            session_cum_score(list): the cumulative score over all trails
    '''
    # We want a list of how the score for every trial and a last score to make it cumulative
    cum_scores = []
    last_score = 0

    # Go through all sessions
    for session in mouse.sessions:
        session_data = mouse.all_data[session]
        total, hits, misses = [0, 0, 0]
        
        # Check if trail was a hit or miss
        for idx, trial in session_data.iterrows():
            total += 1
            if trial['succes']:
                hits += 1
            else:
                misses += 1
                
            # Update the cum. score
            trial_cum_score = (hits - misses) + last_score
            cum_scores.append(trial_cum_score)
        
        # Add the last value of the previous session to make it cumulative
        last_score = trial_cum_score
    return cum_scores

def get_average_cum_score(big_cum_score_list):
    ''' Calculate the average cumulative score

        OUTPUT:
            average_list(np.array), std_list(np.array): avg and std calculated over a list of lists
    '''
    # Create a deepcopy of the original list because we'll be poppin' 'n droppin'
    copy_list = deepcopy(big_cum_score_list)

    # Get the maximal amount of trials that were conducted for each animal
    max_len = np.max([len(cum_score_list) for cum_score_list in copy_list])
    average_list = []
    std_list = []

    # Go through all trials, pop cum. score from their copied list
    for i in range(max_len):
        scores = [cum_score_list.pop(0) for cum_score_list in copy_list if cum_score_list]
        
        # Get standard deviation and average
        std = np.std(scores)
        std_list.append(std)
        average = np.average(scores)
        average_list.append(average)
    return np.array(average_list), np.array(std_list)

def get_blocked_score(original_list, n):
    ''' Cuts a list into blocks of n

        INPUT:
            original_list(list):
            n(int):
        OUPUT:
            list_avg(list?):
    '''
    # Create a queu 
    queue = Queue(maxsize=n)
    list_avg = []

    # Iterate through the list:
    for i in range(len(original_list)):
        # When the que contains more than n values pop the oldest value
        if i >= n:
            queue.get()
            
        # Update the que by adding the next trials value
        queue.put(original_list[i])

        # Get average of queue and append
        list_avg.append(np.mean(queue.queue))
    return list_avg