''' Mouse_Data.py

    Contains the Mouse_Data class that is used for analysing the BPOD data .mat files.
    @Mik Schutte
'''
import numpy as np
import pandas as pd
import os, re, datetime, scipy.io
from scipy.io import matlab
from pymatreader import read_mat

# Issure here with string as integer
def format_data(checked_data):
    '''
    Formats the checked data into a pandas DataFrame
    '''
    # Check for mat objects
    df = pd.DataFrame(columns=['trialType', 'trialStart', 'trialEnd', 'stim_t', 'response_t', 'success', 'licks', 'whisker', 'pupil'])
    df['trialType'] = checked_data['SessionData']['TrialTypes']
    df['trialStart'] = checked_data['SessionData']['TrialStartTimestamp']
    df['trialEnd'] = checked_data['SessionData']['TrialEndTimestamp']
    
    stim_t = [trial['States']['Stimulus'][0] for trial in checked_data['SessionData']['RawEvents']['Trial']]
    df['stim_t'] = checked_data['SessionData']['TrialStartTimestamp'] + stim_t
    
    # The response time should be the first lick after stimulus onset for now.
    # Since the diff in wait for lick calculated the time after stim offset we add 0.2s, to make it correspond with the Stim onset
    # !!!TODO!!!! hardcoded a + 0.2 to the response time 
    response_t = [np.diff(trial['States']['WaitForLick'])[0] + 0.2 for trial in checked_data['SessionData']['RawEvents']['Trial']] # Added 200 ms so that it is the first lick post Stimulus onset (stim_t)
    df['response_t'] = response_t

    # We'll change the RT to the first lick 100 ms after stimulus onset, but how?

    success = [np.isnan(trial['States']['Reward'][0]) for trial in checked_data['SessionData']['RawEvents']['Trial']]
    df['success'] = np.invert(success)    # TODO succes should be capped within response window

    # Licks
    # I dont think this takes in to account at which time the lick was conducted, but we can set this later for when the trial happened with stim_t
    # Go through all trials and get the timestamp for when a lick (Port1 TTL) was detected
    for i in range(len(checked_data['SessionData']['RawEvents']['Trial'])):
        licks = np.array([])
        if 'BNC1High' in checked_data['SessionData']['RawEvents']['Trial'][i]['Events'].keys():
            licks = np.append(licks, checked_data['SessionData']['RawEvents']['Trial'][i]['Events']['BNC1High'])
        if 'BNC1Low' in checked_data['SessionData']['RawEvents']['Trial'][i]['Events'].keys():
            licks = np.append(licks, checked_data['SessionData']['RawEvents']['Trial'][i]['Events']['BNC1Low'])
        df.at[i, 'licks'] = np.array(sorted(licks)) + df.loc[i, 'trialStart'] # Get the absolute time of the lick (not relative to trialStart)
    return df

def get_duplicates(folder, path_ext='microstim'): # Folder likely is root + ID, all files for animal
    ''' Goes through all files in the folder and detects which session had duplicates
        INPUT:
            folder(str): path to the folder, likely root + ID + microstim + Session Data
        OUTPUT:
            duplicate_dict(dict): dictionary with keys being sessions that have duplicates
                                  and values being all the files of that session
    '''
    # Make a file_list containing all files in the folder and get all session names
    file_list = os.listdir(folder)
    sessions = [re.split(f'{path_ext}_', file)[1] for file in file_list] # Remove prefix
    sessions = [re.split('.mat', file)[0] for file in sessions] # Remove .mat extension
    session_date = [re.split('_',session)[0] for session in sessions] # Convert to only the date of the day

    # Go trough all session dates and count the occurance if there are multiple they are duplicate sessions
    duplicates = []
    for session in session_date:
        occurance = session_date.count(session)
        if occurance > 1: 
            duplicates.append(session)
    duplicates = set(duplicates) # This is a set with the session dates of duplicates

    # Create a nested dictionary with keys being a duplicate session and values as dictionary of behaviours
    duplicate_dict = {}
    for session in duplicates:
        duplicate_dict[session] = [file for file in file_list if session in file]
    return duplicate_dict

def check_aborted(trialData):
    ''' Uses all licks during the trial and the stimulus time to determine if no licks were made during the first 100ms of the stimulus.
    
    INPUT:
        trialData (pd.series): single trail components of behaviour
    OUTPUT:
        bool_violate, violations (tup): a boolien stating if lick violations were found
                                        and a list of which times this was.
    '''
    # Unpack variables from the trialData df
    licks = trialData['licks']
    stim_t = trialData['stim_t']

    # If any licks happened during the first 100 ms of the stimulus 
    bool_violate = any((licks >= stim_t) & (licks <= stim_t + 0.1))
    violations = licks[np.where(np.logical_and(licks>=stim_t, licks<=stim_t+0.1))[0]]
    return bool_violate, violations


class Mouse_Data:
    ''' Class designed for housing all data for an individual mouse
        
        INPUT:
            path_to_data(str): path to the mouse folder you want to extract the data from
            
        OUTPUT:
            Mouse_Data(Class): Dataclass with attributes like id, sessions, all_data and concatenated data
    '''    

    def __init__(self, path_to_data, path_ext='microstim'): 
        # From path_to_data get path and files in the raw-folder of that path
        self.path = path_to_data + path_ext + '/Session Data/'
        self.path_ext = path_ext # TODO change name to protocol 
        self.files = os.listdir(self.path)
        self.id = self.files[0].split('/')[-1].split('_')[0]
        self.concat_needed = False
        self.get_behaviour()
        self.sessions = [str(key) for key in self.session_data.keys()]

        if self.concat_needed:
            self.concat_data()

        # self.update_aborted() 
        self.compile_data() # Not necessary because update.aborted also compiles

        if self.concat_needed:
            self.concat_data()

    def get_behaviour(self):
        ''' Creates self.session_data a dictionary with keys being session_dates and values being a pd.Dataframe 
        '''
        self.session_data = {}
        for file in self.files:
            rawData = read_mat(self.path + file)
            session = rawData['__header__'].decode()
            session = re.split('Mon |Tue |Wed |Thu |Fri |Sat |Sun ', session)[-1] 
            session = str(datetime.datetime.strptime(session, '%b %d %X %Y')).split()[0] # It's possible to recover time by not slicing this string or [-1]

            # Format the date as per (day_month_year) format
            date_object = datetime.datetime.strptime(session, "%Y-%m-%d")
            session = date_object.strftime("%d_%m_%Y")

            # Check if a similar session is already in the dictionary
            if session in self.session_data.keys():
                print(f'WARNING: There is already data loaded for the session on {session} of {self.id}.\nData will be concatenated; please check validity.')
                self.concat_needed = True
            self.session_data[session] = format_data(rawData)
    
    def compile_data(self):
        ''' Creates one big pd.DataFrame of all stimuli over all sessions'''
        df_full = pd.DataFrame()
        for session in self.sessions:
            df_full = pd.concat([df_full, self.session_data[session]])
        self.full_data = df_full.reset_index(drop=True)   

    def get_dlc(self, feature, file_end='.h5'): # TODO check if dlc has already been loaded
        ''' For a single Mouse_Data look for DLC files and add it to the .session_data and .full_data

            INPUT:
                mouse_data (Mouse_Data): 
                feature(str): behavioural feature that is tracked by DLC, pupil or whisker
                file_end(str): fileType ending that is used for searching DLC file
            OUTPUT:
                DLCdata(pd.DataFrame): like session_data, but with added raw DLC files for each trial. 
        '''
        # Check input
        if feature not in ['pupil', 'whisker']:
            TypeError(f'{feature} is not known please check spelling.')
        if feature == 'whisker':
            nFrames = 600
        elif feature == 'pupil':
            nFrames = 300

        # Configure path from mouse.id
        root = self.path.split('/Session Data/')[0]
        feature_ext = '/Videos/' + feature + '/'

        # Go through all sessions in Mouse_Data
        for session in self.sessions:
            # Check if the feature was recorded for that sessions
            if session not in os.listdir(root+feature_ext):
                print(f'{self.id} {session} doesnt seem to have any folder with recordings for the {feature} feature.')
                continue

            # Check if there are any DLC files 
            dlcPath = root + feature_ext + session + '/'
            dlc_files = [file for file in os.listdir(dlcPath) if file.endswith(file_end)] 
            if len(dlc_files) == 0:
                print(f'{session} doesnt have any DLC-files.')
                continue

            # Now read the trial DLC data and slice unneccesary scorer off
            for file in dlc_files: 
                trialDLC = pd.read_hdf(dlcPath + file)
                trialDLC = trialDLC[trialDLC.keys()[0][0]]

                # Get the trialNumber from the string
                nTrial = int(file.split('DLC_')[0].split(feature+'_')[1])

                # Check if the DLC file contains the right number of frames TODO likelihood needs to be done for each feature point individually
                dlcSuccess = True
                if len(trialDLC) != nFrames:
                    dlcSuccess = False
                
                # Add the DLC data to the Mouse_Data
                if dlcSuccess:
                    self.session_data[session][feature][nTrial] = trialDLC       

            # Update Mouse_Data.full_data by re-concatenating the sessions 
            self.compile_data()

    def concat_data(self):
        ''' Go through all files, find duplicate sessions and concatenate the files
        
            INPUT:
                folder(str): path to the raw folder that contains the .txt data
            OUTPUT:
                concatenated files: original files have been placed in the raw folder and
                                    an 'old' folder has been added that houses the split data
        ''' 
        # Create a nested dictionary with keys being a duplicate session and values as dictionary of behaviours
        duplicate_dict = get_duplicates(self.path, self.path_ext)
        
        # Go through all duplicate sessions
        for session in duplicate_dict.keys():
            files_to_concatinate = duplicate_dict[session]

            for i, file in enumerate(files_to_concatinate):
                # load in file
                rawData = read_mat(self.path + '/' + file)
                df = format_data(rawData)

                # If a previous df was loaded add the values together
                if i > 0:
                    # Adjust df with values from old_df  trialStart	trialEnd stim_t 
                    # Licktimes of the df are based on BNC input + trialStart we need to remove trialStart again
                    licks = df['licks'] - df['trialStart']

                    endTime = old_df.iloc[-1]['trialEnd'] # Append the sessions together by starting from the end of the last session

                    # TODO this seems convoluted cant I add endTime to all three at the same time
                    df['trialStart'] = df['trialStart'] + endTime
                    df['trialEnd'] = df['trialEnd'] + endTime
                    df['stim_t'] = df['stim_t'] + endTime

                    # Add the new trialStart to the lickTimes
                    df['licks'] = licks + df['trialStart'] # To get the absolute time within the session (not within the trial)

                    df_concat = pd.concat([old_df, df], ignore_index=True)
                old_df = df

            # Format the date as per (day_month_year) format
            date_object = datetime.datetime.strptime(session, "%Y%m%d")
            session = date_object.strftime("%d_%m_%Y")
            self.session_data[session] = df_concat
            self.compile_data()

    def update_aborted(self):
        ''' docstring
        '''  
        for session in self.sessions:
            session_data = self.session_data[session]

            aborted_list = []
            for i, trialData in session_data.iterrows():
                bool_violate = check_aborted(trialData)
                aborted_list.append(bool_violate)
            
            # New column
            session_data['aborted'] = aborted_list

        # Also updata full_data    
        self.compile_data()