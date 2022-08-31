''' Mouse_Data.py

    Contains the Mouse_Data class that is used for analysing the SPIKE2 data .txt files.
    @Mik Schutte
'''
import numpy as np
import pandas as pd
import os, re, datetime

class Mouse_Data:
    ''' Class designed for housing all data for an individual mouse
        
        INPUT:
            path_to_data(str): path to the mouse folder you want to extract the data from
            
        OUTPUT:
            Mouse_Data(Class): Dataclass with attributes like id, sessions, all_data and concatenated data
    '''    

    def __init__(self, path_to_data): 
        # From path_to_data get path and files in the raw-folder of that path
        self.path = path_to_data
        self.files = os.listdir(self.path+'/raw')
        self.id = self.path.split('/')[-1]

        # Run functions that extract data from the files in path_to_data
        self.get_sessions()
        self.get_behaviour()
        self.compile_data()

        
    def get_sessions(self): #TODO probably create failsafe for different month/years
        ''' Create a list of files for individual session dates in self.files
        '''
        # Go through all files
        sessions = []
        for file in self.files:
            if file.endswith('.txt'):
                session = re.split('(?=[A-Z])', file)[0]
                # Get the unique sessions
                if '(' not in session:
                    sessions.append(session)
        # Sort according to date
        sessions = set(sessions)
        sessions_date = sorted([datetime.datetime.strptime(session, '%d_%m_%Y') for session in sessions])
        sessions = [datetime.datetime.strftime(date, '%d_%m_%Y') for date in sessions_date]
        self.sessions = sessions
        

    def get_behaviour(self):
        ''' Creates self.all_data a dictionary with keys being session_dates and values being a pd.Dataframe 
        '''
        self.all_data = dict.fromkeys(self.sessions)
        # For all sessions
        for session in self.sessions:
            # Check if file belongs to session
            for file in self.files:
                if session in file:

                    # Read session data 
                    if 'ReactCalctTXT' in file:
                        RT_df = pd.read_csv(self.path+'/raw/'+file, delimiter='\t', header=None,skiprows=1) #TODO can just be self.path
                    if 'TextMarks' in file:
                        text_df = pd.read_csv(self.path+'/raw/'+file, delimiter='\t', header=None) #TODO can just be self.path
                        text_df = text_df.fillna('NaN')
                        
            # Get the timepoint of intensity change and link intensity to stimtimes
            i_change = text_df.loc[text_df[2].str.contains('Current')]
            data_dict = {'stim_t':[],
                         'response_t':[],
                         'intensity':[],
                         'succes':[]}

            # Go through all stimulus responses
            for idx, stim_t in enumerate(RT_df[0]):
                # Check if the stimulus occured without having a documented intensity
                if stim_t <= np.min(i_change[0]):
                    pass
                # If stim_t happened after an intensity change mark this
                else:
                    exeed = np.where(stim_t > i_change[0])[0][-1]
                    change_idx = i_change.index[exeed]

                    # Get the newly set intensity
                    try:
                        intensity = np.float64(re.split(' |uA',i_change[2][change_idx])[-2].strip('.'))
                    except:
                        break

                    # Put all the seperate DataFrames in a dictionary
                    data_dict['stim_t'].append(stim_t)
                    data_dict['response_t'].append(RT_df[1][idx])
                    data_dict['intensity'].append(intensity)
                    succes = RT_df[1][idx] > 0 and RT_df[1][idx] <= 0.7 and RT_df[1][idx] >= 0.15
                    data_dict['succes'].append(succes)
                
            # Make DataFrame and add to all_data dict
            data_dict = pd.DataFrame(data_dict)  
            self.all_data[session] = data_dict
            
            
    def compile_data(self):
        ''' Creates one big pd.DataFrame of all stimuli over all sessions'''
        df_full = pd.DataFrame()
        for session in self.sessions:
            df_full = pd.concat([df_full, self.all_data[session]])
        self.full_data = df_full
        
