''' Mouse_Data.py

    Contains the Mouse_Data class that is used for analysing the SPIKE2 data .txt files.
    @Mik Schutte
'''
import numpy as np
import pandas as pd
import os, re, datetime, scipy.io
from scipy.io import matlab

def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d
    
    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list, dtype='object')
        else:
            return ndarray

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)

def format_data(checked_data):
    '''
    Formats the checked data into a pandas DataFrame
    '''
    # Check for mat objects
    df = pd.DataFrame(columns=['trialType', 'trialStart', 'trialEnd', 'stim_t', 'response_t', 'success', 'licks'])
    df['trialType'] = checked_data['SessionData']['TrialTypes']
    df['trialStart'] = checked_data['SessionData']['TrialStartTimestamp']
    df['trialEnd'] = checked_data['SessionData']['TrialEndTimestamp']

    stim_t = [trial['States']['Stimulus'][0] for trial in checked_data['SessionData']['RawEvents']['Trial']]
    df['stim_t'] = checked_data['SessionData']['TrialStartTimestamp'] + stim_t

    response_t = [np.diff(trial['States']['WaitForLick'])[0] for trial in checked_data['SessionData']['RawEvents']['Trial']]
    df['response_t'] = response_t

    success = [np.isnan(trial['States']['Reward'][0]) for trial in checked_data['SessionData']['RawEvents']['Trial']]
    df['success'] = np.invert(success)

    # Licks
    for i in range(len(checked_data['SessionData']['RawEvents']['Trial'])):
        licks = np.array([])
        if 'Port1In' in checked_data['SessionData']['RawEvents']['Trial'][i]['Events'].keys():
            licks = np.append(licks, checked_data['SessionData']['RawEvents']['Trial'][i]['Events']['Port1In'])
        if 'Port1Out' in checked_data['SessionData']['RawEvents']['Trial'][i]['Events'].keys():
            licks = np.append(licks, checked_data['SessionData']['RawEvents']['Trial'][i]['Events']['Port1Out'])
        df['licks'].iloc[i] = sorted(licks) # Apperently this is setting with a copy, but I failed to remove this error
    return df

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
        self.files = os.listdir(self.path)
        self.id = self.files[0].split('/')[-1].split('_')[0]
        self.get_behaviour()
        self.sessions = [str(key) for key in self.session_data.keys()]
        self.compile_data()

    def get_behaviour(self):
        ''' Creates self.session_data a dictionary with keys being session_dates and values being a pd.Dataframe 
        '''
        self.session_data = {}
        for file in self.files:
            rawData = load_mat(self.path + file)
            session = rawData['__header__'].decode()
            session = re.split('Mon |Tue |Wed |Thur |Fri |Sat |Sun ', session)[-1] 
            session = str(datetime.datetime.strptime(session, '%b %d %X %Y')).split()[0] # It's possible to recover time by not slicing this string or [-1]
            
            # Check if a similar session is already in the dictionary
            if session in self.session_data.keys():
                print(f'WARNING: There is already data loaded for the session on {session}.\nPlease check validity.')
            self.session_data[session] = format_data(rawData)
    
    def compile_data(self):
        ''' Creates one big pd.DataFrame of all stimuli over all sessions'''
        df_full = pd.DataFrame()
        for session in self.sessions:
            df_full = pd.concat([df_full, self.session_data[session]])
        self.full_data = df_full   