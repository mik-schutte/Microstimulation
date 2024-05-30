''' helpers.py

    Contains functions like get_trial_blocks, get_threshold_data and get_cum_score to analyze the
    Mouse_Data generated through Spike2 input.

    @mik-schutte
'''
import numpy as np
import pandas as pd
from scipy import stats
from queue import Queue
from copy import deepcopy
from Mouse_Data import Mouse_Data

def select_trialType(mouse_data, trialType):
    ''' Slices all trials of a specific type.

        INPUT:
            mouse_data (.full_data, .session_data): the DataFrame containing experimental data
            trialType (int or str): for stim trials 1 or 'stim' for catch 2 or 'catch'.
        OUTPUT:
            sliced mouse_data: the previous input but now for only one trialtype.
    '''
    # Define allowed trialtypes and check input
    allowed_trialTypes = ['test', 1, 'catch', 2]
    if trialType not in allowed_trialTypes:
        raise NameError('trialType not found')
    if not isinstance(mouse_data, pd.DataFrame):
        raise TypeError('mouse_data is not a DataFrame, please select a mouse_data.full_data or mouse_data.session_data[session]')

    # If a string is given change to the numerical code test=1, catch=2
    if trialType == 'test':
        trialType = 1
    elif trialType == 'catch':
        trialType = 2

    # Now only select trials that are of the correct type
    typeData = mouse_data.loc[mouse_data['trialType'] == trialType]
    return typeData


def get_FirstLick_idx(trialData):
    '''docstring'''
    # Unpack the licks and the stimulus time from the pd.Series
    stim_t = trialData['stim_t']
    licks = trialData['licks']

    # Check if there where any licks during this trial
    if licks.size == 0:
        return False
    
    # Now check if there was a lick after the stimulus at all 
    postLicks = np.where(licks >= stim_t+0.1)[0] #Has to add +0.1 so that we are not counting licks that abort the trial
    if len(postLicks) == 0:
        return False
    else: 
        iFirstLick = postLicks[0]
    return iFirstLick


def check_lickPause(trialData): #So far this works every time (i.e. is not lower than 3s)
    '''Checks the time between the last lick before the stimulus. There should be at least 3s in between.
    '''
    # Get the index of the first lick after stimulus
    iFirstLick = get_FirstLick_idx(trialData)

    # If this iFirstLick was False than there were either no licks, or no licks post stimulation
    if iFirstLick:
        licks = trialData['licks']
        lickPause = licks[iFirstLick] - licks[iFirstLick-1] # Check the difference between the first lick post stim and the last lick before
        return lickPause
    else:
        return False


def curateLicks(trialData):
    ''' Conduct pre-analysis concerning the responsetime when it is too close after stim onset. 	

        Basically aligns the licks to the stimulus time
    
        NOTE: This will bess with the analysis of DLC data because there the animal receives water based on the start of the Reward State
    '''
    # Unpack some values for ease of usage
    response_t = trialData['response_t']
    licks = trialData['licks']
    stim_t = trialData['stim_t']

    # Check if the first lick occured 100 ms after the stimulus onset
    # If licks were given before the end of the Stimulus State it won't be the same as the response_t
    # This is because response_t is determined based on the beginning and end time of the WaitForLick state
    # iFirstLick is based on the first lick after stim offset
    iFirstLick = get_FirstLick_idx(trialData)

    if iFirstLick: 
        curatedLicks = licks[iFirstLick:] - stim_t
    elif iFirstLick == False:
        curatedLicks = np.array([])
    return curatedLicks


def check_aborted(trialData):
    ''' Uses all licks during the trial and the stimulus time to determine if no licks were made during the first 100ms of the stimulus.
    '''
    # Unpack variables from the trialData df
    licks = trialData['licks']
    stim_t = trialData['stim_t']

    # If any licks happened during the first 100 ms of the stimulus 
    violations = np.where(np.logical_and(licks>=stim_t, licks<=stim_t+0.1))
    
    # If there were violations than the length should be larger than zero
    bool_violate = np.bool(np.size(violations))
    return bool_violate


def get_PLick(mouse, catchInf=False):
    ''' Calculates the chance of a lick during stim and catch trials for oall sessions

    INPUT:
        mouse (Mouse_Data class): 
        catchInf (bool): If True you will take into account 100% hits or misses

    OUTPUT:
        mstimP_array, catchP_array (tuple): tuple of arrays of the chance to lick during stim and catch trials.
    '''
    # Hold the P(lick) over sessions in a list
    mstimP_array = []
    catchP_array = []
    # Go through all training sessions and calculate the hit chance during mStim and catch
    for session in mouse.sessions:
        session_data = mouse.session_data[session]
        mstim = select_trialType(session_data, 'test')
        catch = select_trialType(session_data, 'catch')

        # Now for the total of mstim and catch trials determing how many were a hit for that session
        mstim_hit = mstim.loc[mstim['success'] == True]
        catch_hit = catch.loc[catch['success'] == True]

        # Now calculate the percentage of total trialTypes per session
        mstimP = len(mstim_hit)/len(mstim)
        catchP = len(catch_hit)/len(catch)

        # Catch infinite values for d' calculation
        if catchInf:
            if mstimP == 1:
                mstimP = (1-1/(2*len(mstim)))
            elif mstimP == 0:
                mstimP = (1/(2*(len(mstim))))
            if catchP == 1:
                catchP = (1-1/(2*len(catch)))
            elif catchP == 0:
                catchP = (1/(2*len(catch))) 

        # Add the chance values to the array
        mstimP_array.append(mstimP)
        catchP_array.append(catchP)
    return mstimP_array, catchP_array


def calc_d_prime(mouse_data):
    ''' Calculates the d' (Sensitivity index) for each session of mouse_data

    INPUT: 
        mouse_data (Class):
    OUTPUT:
        d_prime_list (list): list of the d'value for each session
    '''
    d_prime_list = []
    # Get P(lick) for catch and mStim trials for all sessions
    mStim_Plicks, catch_Plicks = get_PLick(mouse_data, catchInf=True)

    # For each session get the P(lick) for mStim and catch trails
    for mStim_Plick, catch_Plick in zip(mStim_Plicks, catch_Plicks):
        # Calculate the inverse of the cdf (ppf) of each P-value
        mStim_z = stats.norm.ppf(mStim_Plick)
        catch_z = stats.norm.ppf(catch_Plick)

        # Now calculate d' = |z(mStim hit) - z(catch hit)|
        d_prime_list.append(abs(mStim_z - catch_z))
    return d_prime_list


def get_hitnmiss(mouse_data):
    ''' Returns the total number of hits and misses in the dataframe

    INPUT:
        mouse_data (pd.Dataframe): either .full_data or session_data
    OUTPUT:
        nHits, nMisses (tuple): total number of hits and misses
    '''
    # Check input
    if not isinstance(mouse_data, pd.DataFrame):
        raise TypeError('mouse_data is not a DataFrame, please select a mouse_data.full_data or mouse_data.session_data[session]')
    
    # Select hits (correct trials) and misses (incorrect trials)
    hits = mouse_data.loc[mouse_data['success'] == True]
    misses = mouse_data.loc[mouse_data['success'] == False]
    return len(hits), len(misses)


def calculate_response_times(ctrl_data, trial_type):
    '''Calculate average response times for a given trial type across sessions. 

    INPUT:
        TODO ctrl_data is a list of mouse_data classes to calculate the average rt over.
        
    ''' 
    rt_data = []
    for mouse in ctrl_data:
        rt_indi = []
        for session_name in mouse.sessions:
            session = mouse.session_data[session_name]
            trials = select_trialType(session, trial_type)
            trials = trials.loc[trials['success'] == True]
            rt = np.average(trials['response_t'])
            rt_indi.append(rt)
        rt_data.append(rt_indi)
    return rt_data



# LEGACY OLD CODE FOR WHEN WE STILL USED THE DECREASING STIM THRESHOLD
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


def get_avg_std_threshold(threshold_data, max_sessions=5): #TODO rename this function as it works for everything
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
        session_data = mouse.session_data[session]
        total, hits, misses = [0, 0, 0]
        
        # Check if trail was a hit or miss
        stimTrials = select_trialType(session_data, 1)
        for idx, trial in stimTrials.iterrows():
            if idx >= 150: # Note this should not be hardcoded
                break
            total += 1
            if trial['success']:
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
    # TODO check if lists are of same length or if wre need to extend th elists

    # Get the maximal amount of trials that were conducted for each animal
    max_len = np.max([len(cum_score_list) for cum_score_list in copy_list]) # Should be at 150 trials
    
    data = {'raw': copy_list, 'avg': np.mean(copy_list), 'med':np.median(copy_list), 'std':np.std(copy_list) , 'sem':np.std(copy_list)/len(copy_list)} #parametrics

    average_list = []
    median_list = []
    std_list = []
    sem_list = []

    # Go through all trials, pop cum. score from their copied list
    for i in range(max_len):
        scores = [cum_score_list.pop(0) for cum_score_list in copy_list if cum_score_list]
        
        # Get standard deviation and average
        average = np.average(scores)
        average_list.append(average)
        median = np.median(scores)
        median_list.append(median)
        std = np.std(scores)
        std_list.append(std)
        sem = std / len(big_cum_score_list)
        sem_list.append(sem)

        # Add median, SEM
    data = {'raw': big_cum_score_list, 'avg':average_list, 'std': std_list, 'med':median_list, 'sem':sem_list}
    return data


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