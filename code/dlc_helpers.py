'''
'''

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, re, math 
from copy import deepcopy
from queue import Queue
from Microstimulation.Mouse_Data import Mouse_Data

def get_feature_dlc_data(folder_path, feature, file_end='.h5'):
    ''' Extracts the DLC files of a specific behavioural feature whisker or pupil from a folder path
    
        INPUT: 
            folder_path(str): path the the folder that contains the dlc files ...ID/20-07-2022
            feature(str): feature you want to select: left_whisker, right_whisker or pupil
            file_end(str) the extenstion of your dlc file-type DEFAULT=.h5
        OUTPUT:
            succes(bool): did it work
            dlc_data(pd.DataFrame): a pandas DataFrame of all marker x, y and likelihood values of a feature 
    '''
    # Quick check of feature
    allowed_features = ['left_whisker', 'right_whisker', 'pupil']
    if feature not in allowed_features:
        raise AttributeError(f'Feature: {feature} not found, try {allowed_features}.')
    elif 'whisker' in feature:
        feature_id = feature.split('_')[0] # Gets a string to identify the correct dlc files by feature (e.g. left or right)
    else:
        feature_id = feature # For pupil its just pupil
        
    folder_path = folder_path + '/prepped/'
    
    # Now get all files the dlc output file type
    folder_files = os.listdir(folder_path)
    dlc_files = [file for file in folder_files if file.endswith(file_end)] 
    
    # Check if the feature has data and extract the data of a specific feature from folders of dlc_files
    try: 
        feature_file = [file for file in dlc_files if feature_id in file][0]
        feature_data = pd.read_hdf(folder_path + feature_file)
        succes = True
    except:
        feature_data = None
        succes = False
    return feature_data, succes


def get_dlc_dict(mouse):
    ''' Creates a dictionary with keys being session and value being a dictionary with keys being the tracked component
        (e.g. left_whisker, right_whisker, pupil) and the value being the path of the DLC-analyzed .h5 file for a single mouse
        
        INPUT:
            mouse(Mouse_Data): Dataclass with attributes like id, sessions, all_data and concatenated data
        OUTPUT:
            dlc_dict(dict): nested dictionary containing the DLC results, with keys being session and tracked features
    '''
    # Create a nested dictionary with session as key and a dict as value
    dlc_dict = {}
    path = mouse.path
    features = ['left_whisker', 'right_whisker', 'pupil']
    
    # For every session try to get all feature dlc files
    for session in mouse.sessions:
        dlc_dict[session] = {}
        session_folder = session.replace('_','-')
        folder_path = path +'/' + session_folder #_map + '/prepped/'
#         folder_files = os.listdir(folder_path)
#         dlc_files = [file for file in folder_files if file.endswith('.h5')] # My DLC output is in .h5 format, therefore I only select these
        
        for feature in features:
            feature_data, succes = get_feature_dlc_data(folder_path, feature, file_end='.h5')
            # Check if this worked
            if succes:
                feature_key = feature_data.keys()[0][0]
                dlc_dict[session][feature] = feature_data[feature_key]
            else:
                dlc_dict[session][feature] = feature_data
                
    return dlc_dict #BRO PLEASE PUT THIS IN THE MOUSE_DATACLASS


def get_x_centered(response_t, fps, frame_count):
    '''docstring'''
    # From fps get frame duration
    T = 1/fps
    # Round response time to 5 ms
    response_t_round = round(round(response_t/T, 0)*T, 3)
    # Link frame index to response time
    response_t_frame = int(response_t_round*fps)
    # Get array where 0 indicates the frame of response
    x_centered = np.arange(0, frame_count, 1) - response_t_frame
    return x_centered

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

def normalize_whisking(whisker_coordinates, ll_threshold=0.95, n_avg=5):
    ''' Min-max normalizes the coordinates of whisker feature.
    
        INPUT: 
            whisker_coordinates(dlc_dict): dictionary containing the x and y coordinates and likelihood of the feature
            ll_threshold(float) minimum likelihood for accepting the maximal position
            n_avg(int): group size for determining extreme whisker position
        OUTPUT:
            normalized_whisking(np.array):
    '''
    raw_y = whisker_coordinates['y']
    ll = whisker_coordinates['likelihood']
    
    # Only take coords into account with sufficient likelihood of estimation
    checked_coordinates = whisker_coordinates.loc[ll>=ll_threshold]
    y_cords = checked_coordinates['y']

    # Get the coordinates of the 5 most retracted lowest y and most protracted highest y positions
    retracted_coords = np.mean(y_cords.nsmallest(n_avg))
    protracted_coords = np.mean(y_cords.nlargest(n_avg))
    
    # Perform min-max normalization
    normalized_coords = (raw_y - retracted_coords) / (protracted_coords - retracted_coords)
    return normalized_coords

def normalize_whisking_z(whisker_coordinates, ll_threshold=0.95, n_avg=5):
    ''' Min-max normalizes the coordinates of whisker feature.
    
        INPUT: 
            whisker_coordinates(dlc_dict): dictionary containing the x and y coordinates and likelihood of the feature
            ll_threshold(float) minimum likelihood for accepting the maximal position
            n_avg(int): group size for determining extreme whisker position
        OUTPUT:
            normalized_whisking(np.array):
    '''
    raw_y = whisker_coordinates['y']
    ll = whisker_coordinates['likelihood']
    
    # Only take coords into account with sufficient likelihood of estimation
    checked_coordinates = whisker_coordinates.loc[ll>=ll_threshold]
    y_cords = checked_coordinates['y']

    # Get the coordinates of the 5 most retracted lowest y and most protracted highest y positions
    retracted_coords = np.mean(y_cords.nsmallest(n_avg))
    protracted_coords = np.mean(y_cords.nlargest(n_avg))
    
    # Perform min-max normalization
    normalized_coords = (raw_y - retracted_coords) / (protracted_coords - retracted_coords)
    return normalized_coords


def average_whisker_movement(session_data, normalized_trace, window=250):
    ''' Calculates the average whisker movement centered around the response or stimulation onset
        INPUT:
            session_data(pd.DataFrame)
            normalized_traces(array): unchunked
            window(int):
        OUTPUT:
            avg_whisker_trace(array):
    '''
    # Some basics for whisker video
    frame_count = 255
    fps = 200
    T = 1/fps
    window_size = (window/1000) / T
    
    # Average only succesful and unsuccesful data dont do both but implement a check for it? nah
    # Split into succesful and unsuccesful trials 
    succes_trials = session_data.loc[session_data['succes']==True]
    fail_trials = session_data.loc[session_data['succes']==False]
    
    chunked_traces = chunks(normalized_trace, frame_count)
    succes_traces = []
    fail_traces = []
    x_list = []
    
    # Select the traces of all succesful trials and get average
    for i in succes_trials.index:
        response_t = succes_trials['response_t'].loc[i]
        x_centered = get_x_centered(response_t, fps, frame_count) * T
        
        # Try to window trace around response
        stim_idx = np.argwhere(x_centered==0)[0][0]
        min_window = int(stim_idx - window_size)
        max_window = int(stim_idx + window_size)
        windowed_trace = chunked_traces[i][min_window:max_window]
        if windowed_trace:
            succes_traces.append(windowed_trace)
            x_list.append(x_centered)
        else:
            pass
#             print(f'Window not availible for succesful trial#{i}')
        
    for i in fail_trials.index:
        fail_traces.append(chunked_traces[i])
    
    succes_avg, succes_std = get_average_cum_score(succes_traces)
    fail_avg, fail_std = get_average_cum_score(fail_traces)
    return (succes_avg, succes_std), (fail_avg, fail_std)


#TODO only centered on response now & centered around lick
#TODO additional step before saving, ask user input
def plot_session_traces(mouse, session, marker, plots_per_fig=10, peak=True, save=False, destfolder=None): 
    ''' Plot the marker position for all trials in n session of the mouse                       
        
        INPUT:
            mouse(Mouse_Data):
            session(str): date of experimental session e.g. 01_05_1998
            marker(tuple):
            plots_per_fit(int):
            save(bool): 
            destfolder(str):
            
        OUTPUT: TODO ONLY DOES ONE SESSION RIGHT NOW!!!
    '''
    # Organize which DLC files belong to mouse
    dlc_dict = get_dlc_dict(mouse)
    session_data = mouse.all_data[session]
    
    # TODO should definitly create a checker for frame number and trialnumber to check correct dlc

    # Set normalization, nframes and fps according to marker
    if 'whisker' in marker[0]:
        raw_traces = dlc_dict[session][marker[0]][marker[1]]
        normalized_traces = normalize_whisking(raw_traces, n_avg=5)
        # Now chunk the raw traces into sets of frames for each trial
        chunked_traces = chunks(normalized_traces, nframes)
        y_title = 'Normalized whisker position'
        nframes = 255
        fps = 200
    elif 'pupil' in marker[0]:
        raw_traces = get_pupil_size(dlc_dict[session])
        chunked_traces = normalize_pupil_size(raw_traces, n_avg=5)
        y_title = 'Normalized pupil dilation'
        nframes = 130 #TODO double check this value
        fps = 100 #TODO double check this value
    else:
        print('Unknown number of frames in the burst capture for this marker')
        return
    
    # Check if save and get filename
    if save:
        if not destfolder:
            destfolder = mouse.path+'/'+marker[0]+'_traces/'
            print(f'No destfolder specified, saving to {destfolder}')
        os.makedirs(destfolder, exist_ok=True)
        os.makedirs(destfolder+mouse.id, exist_ok=True)
        fname = session+marker[0]+'_'+marker[1]+'.pdf'
        pdf = PdfPages(destfolder+mouse.id+'/'+fname)    
              
    # Determine how many figures and plots we'll need
    if peak:
        total = plots_per_fig
    else:
        total = len(session_data)
    nfigs = math.ceil(total/plots_per_fig)
    trial_idx = 0
                  
    # For every figure
    for fig_i in range(nfigs):
        # Set fundamentals
        fig, axs = plt.subplots(plots_per_fig, 1, figsize=(15,15))
        fig.text(0.04, 0.5, y_title, va='center', rotation='vertical')
        fig.text(0.5, 0.04, 'Time(s)', ha='center')
        fig.suptitle(session, y=0.925)
        plt.subplots_adjust(hspace=0.5,)
        fig.patch.set_facecolor('white')            
        
        # Fill up figure
        for ax_idx in range(plots_per_fig):
            # Get RT and trace
            try:
                trial = session_data.loc[trial_idx]
                response_t = trial['response_t']
                trace = chunked_traces[trial_idx]
            except:
                break
            
            # Calculate x_centered 
            frame_count = len(trace)
            x_centered = get_x_centered(response_t, fps, frame_count) * (1/fps) # But its -0.1
            
            # Check success
            succes = trial['succes']
            if succes:
                succes_c = 'green'
                axs[ax_idx].axvline(0, color='black', linestyle='--') # Line indication response
                axs[ax_idx].set_xlim([-0.5, 0.8])

            else:
                succes_c = 'red'

            # Plot
            for i in [15, 16, 17, 18]: # range because of not allowed lick here
                axs[ax_idx].axvline(x_centered[i], color='gray')
            axs[ax_idx].plot(x_centered, trace, color=succes_c)
#             axs[ax_idx].set_ylim([0, 1])
            axs[ax_idx].set_ylabel(f'Trial {trial_idx}')
            trial_idx += 1      
            
            # Add legend for the first axs
            if fig_i == 0 and ax_idx == 0:
                red_line = matplotlib.lines.Line2D([], [], color='red', label='Incorrect trial')
                green_line = matplotlib.lines.Line2D([], [], color='green', label='Correct trial')
                stim_line = matplotlib.lines.Line2D([], [], color='gray', label='Stimulus')
                response_line = matplotlib.lines.Line2D([], [], color='black', linestyle='--', label='Response')
                axs[ax_idx].legend(bbox_to_anchor=(0., 1.1, 1., .102), handles=[green_line, red_line, stim_line, response_line],
                                  mode="expand", borderaxespad=0., ncol=2)    
        # Save
        if save:
            pdf.savefig(fig) 
         # Show whole figure     
        else:
            plt.show()
    if save:
        pdf.close()
    plt.clf()
    return        


def get_pupil_size(dlc_session_dict):
    ''' docstring
    '''
    # Lets take what's necessary
    pupil_df = dlc_session_dict['pupil']
    pupil_markers = list(set([marker[0] for marker in pupil_df if 'pupil' in marker[0]]))
    pupil_markers.sort()
    
    # Now get the coordinates for every pupil marker 
    cord_dict = dict.fromkeys(pupil_markers, None)
    for marker in pupil_markers:
        marker_ll = pupil_df[marker]['likelihood'].to_numpy()
        marker_x = pupil_df[marker]['x'].to_numpy()
        marker_y = pupil_df[marker]['y'].to_numpy()
        marker_cords = [np.array((x, y, ll)) for x, y, ll in zip(marker_x, marker_y, marker_ll)]
        cord_dict[marker] = marker_cords   

    # Compare the opposing pupil markers with eachother
    compare_zip = [zip(cord_dict['pupil1'], cord_dict['pupil4']),
                   zip(cord_dict['pupil2'], cord_dict['pupil5']),
                   zip(cord_dict['pupil3'], cord_dict['pupil6'])]
    #TODO ADD A LIKELIHOOD CHECK AND 
    # Get the coordinates and calculate the Euclidian distance in every frame for all opposing markers
    all_distances = []
    all_ll = []
    for comparison in compare_zip:
        distance_all_frames = []    
        ll_all_frames = []
        for cords_i, cords_j, in comparison:
            ll = np.mean([cords_i[2], cords_j[2]])
            ll_all_frames.append(ll)
            distance = np.linalg.norm(cords_i[0:2]-cords_j[0:2])
            distance_all_frames.append(distance)

        all_ll.append(ll_all_frames)
        all_distances.append(distance_all_frames)
    
    pupil_size = pd.DataFrame({'pupil_size':np.median(all_distances, axis=0), 'likelihood':np.mean(all_ll, axis=0)})    
    return pupil_size

def normalize_pupil_size(pupil_size_df, ll_threshold=0.95, n_avg=5, nframes=130):
    ''' Min-max normalizes the size of the pupil feature TODO normalize only with high likelihood distances maybe in get_pupil_size?
    
        INPUT:
            pupil_size_dict(dict): OUTPUT OF get_pupil_size median Euclidian distance between opposing pupil markers for every frame and average likelihood
            n_avg(int): group size for determining extreme pupil size
        OUTPUT:
            normalized_pupil_sizes(list): Chunked normalized between the maximum and minumum whisker dilation
    '''
    # Get the pupil size and likelihood from dict
    pupil_sizes = pupil_size_df['pupil_size']
    pupil_ll = pupil_size_df['likelihood']
    normalized_pupil_sizes = []
    
    # Chunk session frames into trial frames
    trial_pupil = chunks(pupil_sizes, nframes)
    trial_ll = chunks(pupil_ll, nframes)
    
    for trial_sizes, trial_ll in zip(trial_pupil, trial_ll):
        trial_df = pd.DataFrame({'pupil_size':trial_sizes, 'likelihood':trial_ll})
        
        # Snip into the pre-stimulus frames
        # Frame-aquisition started 75 ms before stimulation at 100 fps up to 130 frames: 75/(1/fps)=7.5 --> 8
        pre_trial_df = trial_df[0:8]
        
        # Check for likelihood
        checked_sizes_pre = pre_trial_df.loc[pre_trial_df['likelihood']>=ll_threshold]['pupil_size']
        checked_sizes = trial_df.loc[trial_df['likelihood']>=ll_threshold]['pupil_size']
        
        # Get statistics for z-score
        mean = np.mean(checked_sizes_pre)
        std = np.std(checked_sizes)
        
        # Calculate z-scored pupil size for all frames in trial
        z_scores = (np.array(trial_sizes) - mean) / std
        normalized_pupil_sizes.append(z_scores)
        
#         contracted = np.median(checked_sizes.nsmallest(n_avg))
#         dilated = np.median(checked_sizes.nlargest(n_avg))
        
#         # Perform min-max normalization
#         normalized_trial_sizes = (np.array(trial_sizes) - contracted) / (dilated - contracted)
#         normalized_pupil_sizes.append(normalized_trial_sizes)
    
    return pd.Series(list(normalized_pupil_sizes))


def check_dlc(mouse, feature='whisker'):
    dlc_dict = get_dlc_dict(mouse)
    correct_sessions = []
    for session in mouse.sessions:
        session_data = mouse.all_data[session]
        nstim = len(session_data)
        
        # If no files was read then dont continue
        if feature == 'whisker':
            if isinstance(dlc_dict[session]['left_whisker'], pd.DataFrame):
                nframes = len(dlc_dict[session]['left_whisker']['whisker_tip']['y'])
                whisker_traces = chunks(dlc_dict[session]['left_whisker']['whisker_tip']['y'], 255)
                ntrace = len(whisker_traces)
                if abs(nstim-ntrace) >= 1: #REMOVE EQUAL TO IF BELOW IS UNCOMMENTED
                    print(f'{mouse.id} {session} {feature} stimuli and frames dont correspond. nstim={nstim}, ntrace={ntrace}')
#                 elif abs(nstim-ntrace) == 1:
#                     print(f'{mouse.id} {session} {feature} stimuli and frames dont correspond. nstim={nstim}, ntrace={ntrace} dropping last trail')
#                     session_data.drop(session_data.tail(1), inplace=True)
                else:
                    correct_sessions.append(session)
                    
        elif feature == 'pupil':
            if isinstance(dlc_dict[session]['pupil'], pd.DataFrame):
                pupil_size_df = get_pupil_size(dlc_dict[session])
                nframes = len(pupil_size_df)
                pupil_traces = chunks(pupil_size_df, 130)
                ntrace = len(pupil_traces)
                if abs(nstim-ntrace) >= 1:
                    print(f'{mouse.id} {session} {feature} stimuli and frames dont correspond. nstim={nstim}, ntrace={ntrace}')
#                 elif abs(nstim-ntrace) == 1:
#                     print(f'{mouse.id} {session} {feature} stimuli and frames dont correspond. nstim={nstim}, ntrace={ntrace} dropping last trail')
#                     session_data.drop(session_data.tail(1), inplace=True)
                else:
                    correct_sessions.append(session)
        else:
            print(f'Incorrect feature:{feature}')
    return correct_sessions


def extend_lists(all_lists, max_len=0):
    '''docstring
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
        try:
            last_value = l[-1]
        except:
            print(l)
        l.extend([last_value]*fill_after)
        extended_lists.append(l)
    return extended_lists


def chunks(lst, n):
    '''Yield succesive n-sized chunks from list'''
    chunky_list = []
    for i in range(0, len(lst), n):
        chunk = list(lst[i:i+n])
        chunky_list.append(chunk)
    chunky_list = extend_lists(chunky_list, n)
    return chunky_list

