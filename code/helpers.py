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
            session_data(pd.DataFrame): dataframe containing data from one individual session e.g. Mouse_Data.all_data[sessionID]   
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
            session_data(pd.DataFrame): session_data(pd.DataFrame): dataframe containing data from one individual session e.g. Mouse_Data.all_data[sessionID]
            min_score(float): minimal fraction of succesful trials in the trial block to be a succesful intensity
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
    try: 
        threshold = np.min(list(threshold_dic.keys()))  
        return threshold 
    except:
        print(f'Failed at block {block}')

def get_threshold_list(mouse, min_score):
    ''' Creats a list of threshold, i.e. lowest succesful intensity block over sessions
    
        INPUT:
            mouse(Mouse_Data): Dataclass with attributes like id, sessions, all_data and concatenated data
            min_score(float): minimal fraction of succesful trials in the trial block
        OUTPUT:
            threshold_list(list): values of the lowest succesfully trial for each session
    '''
    # 160 is the initial threshold as this is the naive succesfully detected intensity
    threshold_list = [160] 
    for session in mouse.sessions:
        session_data = mouse.all_data[session]
        threshold = get_threshold(session_data, min_score)
        threshold_list.append(threshold)
        
        # Get if expert already achieved
        if threshold <= 20:
            break
    return threshold_list

def get_threshold_data(mouse_list, min_score):
    ''' Creates threshold_lists for all individuals in a list
    
        INPUT:
            mouse_list(list): list of Mouse_Data classes with attributes like id, session and all_data
            min_score(float): minimal fraction of succesful trials in the trial block
        OUTPUT:
            threshold_data(list): a nested list, with each list containing threshold values for all sessions of one mouse
    '''
    # Lets get all the threshold lists for the control animals
    threshold_data = []
    for mouse in mouse_list:
        threshold_list = get_threshold_list(mouse, min_score)
        # Fill it up to a certain size for nice plotting NOTE: if training took more than the counter (5 days) it wont be added to list
        counter = len(threshold_list)
        while counter <5: 
            threshold_list.append(threshold_list[-1])
            counter = len(threshold_list)
        threshold_data.append(threshold_list)
    return threshold_data 

def get_avg_std_threshold(threshold_data, max_sessions=5):
    ''' Calculate the average threshold and its standard deviation for each session over a list of threshold lists
    
        INPUT:
            threshold_data(list): a nested list, with each list containing threshold values for all sessions of one mouse
            max_sessions(int): maximum number of days the avg and std is calculated over
        OUTPUT:
            avg_list(list): list containing the average threshold for each session
            std_list(list): list containing the standard deviation of the averge threshold for each session
    '''
    avg_list = []
    std_list = []
    for i in range(max_sessions):
        day_list = [threshold[i] for threshold in threshold_data]
        avg = np.mean(day_list)
        avg_list.append(avg)
        std = np.std(day_list) 
        std_list.append(std)
    return avg_list, std_list

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
    ''' Calculate the average cumulative score and its standard deviation over a list of individual scores
    
        INPUT:
            big_cum_score_list(list): nested list of cumulative scores, where each individual list is the cum. score of one mouse

        OUTPUT:
            average_list(np.array): average cumulative score calculated over a list of cumulative score lists
            std_list(np.array): standard deviation of the averague cumulative score
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
    ''' Iterate through a list, block values by n and calculate the average of that block

        INPUT:
            original_list(list): the list you want to get the average of a block from
            n(int): blocksize
        OUPUT:
            list_avg(list): list of block-averaged values
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


def get_average_session_len(mouse_list):
    ''' Calculates the average and standard deviation of session length (i.e. number of trials per session) 
    
            INPUT:
                mouse_list(list): list of Mouse_Data with attributes like id, session and all_data
            OUTPUT:
                len_df(pd.DataFrame): Dataframe containing a multitude of data concerning the session length
   '''
    len_dict = {'raw': {1:[], 2:[], 3:[], 4:[], 5:[]},
                'avg': 0,
                'med': 0,
                'std': 0,
                'sum_avg': {1:0, 2:0, 3:0, 4:0, 5:0},
                'sum_med': {1:0, 2:0, 3:0, 4:0, 5:0}}

    for mouse in mouse_list:
        session_list = []
        for idx, session in enumerate(mouse.sessions):
            session_len = len(mouse.all_data[session])
            len_dict['raw'][idx+1].append(session_len)
            
    len_df = pd.DataFrame.from_dict(len_dict)
    len_df.index.name = 'day'
    previous_sum = 0
    # Now add the avg and std of days
    for day in len_df.index:
        # Check raw contains values
        if len_df.loc[day, 'raw']:
            len_df.loc[day, 'avg'] = np.mean(len_df['raw'][day])
            len_df.loc[day, 'med'] = np.median(len_df['raw'][day])
            len_df.loc[day, 'std'] = np.std(len_df['raw'][day])
            
        if day != 1:
            len_df.loc[day, 'sum_avg'] = len_df['avg'][day] + len_df['sum_avg'][day-1]
            len_df.loc[day, 'sum_med'] = len_df['med'][day] + len_df['sum_med'][day-1]
        else:
            len_df.loc[day, 'sum_avg'] = len_df['avg'][day]
            len_df.loc[day, 'sum_med'] = len_df['med'][day]
    return len_df


def extend_lists(all_lists, max_len=0):
    ''' Extend multiple lists with their last value up to the length of the largest list or a given length
    
        INPUT:
            all_lists(list): nested list of lists you want to extend
            max_len(int): the length of the list you want to have by extending
        OUTPUT:
            extended_lists(list): nested list of extended lists
    '''
    # Create a deepcopy
    copy_lists = deepcopy(all_lists)
    
    # Get max length
    if max_len == 0:
        max_len = np.max([len(l) for l in all_lists])
    
    # Extend with which value and how many
    extended_lists = []
    for l in copy_lists:
        l = list(l)
        fill_after = abs(len(l) - max_len)
        last_value = l[-1]
        l.extend([last_value]*fill_after)
        extended_lists.append(l)
    return extended_lists
