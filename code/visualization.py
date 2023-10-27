''' visualization.py

    Contains functions needed for data visualization, such as plot_raster_rt.

    @mik-schutte
'''
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helpers import *
from scipy.optimize import curve_fit 
matplotlib.rcParams.update({'font.size':16, 'font.family':'Times New Roman', 'axes.facecolor':'white'})   

def plot_raster_rt(mouse_data, save=False, peak=False):
    ''' Creates a figure containing rasterplots of the trial response time.

        INPUT:
            mouse_data(Mouse_Data): Dataclass with attributes like id, sessions,.session_data and concatenated data
            save(bool): prompts the user for destination folder path
            peak(bool): if true only two session rasterplots will be created
        OUTPUT:
            raster_rt_plot(matplotlib.plt): either a plot is shown or saved
    '''
    # Check for peaking allowing the user to only see the plots of the first 2 sessions
    if peak:
        n_sessions = 2
    else:
        n_sessions = len(mouse_data.sessions)
    # Set figure basics 
    fig, axs = plt.subplots(1, n_sessions, figsize=(15, 10)) # Size plot according to the number of sessions
    plt.subplots_adjust(wspace=1.) 
    fig.patch.set_facecolor('white')
    fig.suptitle(str(mouse_data.id), y=1.05)
    
    # Create legend patches
    gray_patch = matplotlib.patches.Patch(color='gray', label='Stimulus')
    orange_patch = matplotlib.patches.Patch(color='orange', label='Catch trials')
    red_patch = matplotlib.patches.Patch(color='red', label='Incorrect trials')
    green_patch = matplotlib.patches.Patch(color='green', label='successful trials')
    
    # Get and plot data for every session
    for idx, session in enumerate(mouse_data.sessions):
        if peak and idx == n_sessions:
            break
        colors = []
        
        # For pairing only pair and mix data are important
        # if catch:
        select_data = mouse_data.session_data[session].loc[(mouse_data.session_data[session]['trialType'] == 2)|(mouse.session_data[session]['trialType'] == 1)|(mouse.session_data[session]['trialType'] == 'pairData')]        
        patches = [gray_patch, green_patch, red_patch, orange_patch]
        
        # Aquire response time, xticks and datatype
        rt_full = select_data['response_t']
        x = np.arange(0, len(rt_full), 1) # Initiate x-axis for plotting
        rt = [[rt] for rt in rt_full] # plt.eventplot requires values as list to ascribe different offsets
        dtype = [[dtype] for dtype in select_data['trialType']]

        # Pick right color
        for _, trial in select_data.iterrows():
            trial_success = trial['success']
            if trial['trialType'] == 2:
                c = 'orange'
            elif trial['trialType'] == 1:
                if trial_success:
                    c = 'green'
                else:
                    c = 'red'
            elif trial['trialType'] == 'pairData':
                c = 'blue'
            colors.append(c)
            
        # Now the plot
        # If there are multiple sessions
        if len(mouse_data.sessions) > 1:
            for x in np.arange(0, 0.15, 0.001):
                axs[idx].axvline(x, color='gray')
            offset = np.arange(0, len(rt), 1)
            axs[idx].eventplot(rt, lineoffsets=offset, linewidth=7.5, colors=colors)
            axs[idx].set_xlim([-0.2, 1.55])
            axs[idx].set_ylabel('Trial #')
            axs[idx].invert_yaxis()
            axs[idx].set_xlabel('Response time (s)')
            axs[idx].set_title(str(session))
            axs[idx].set_xticks(np.arange(0, 1.55, 0.5))
            axs[0].legend(bbox_to_anchor=(0., 1.15, 1., .02), handles=patches, mode="expand", borderaxespad=0., ncol=1)
        
        # If there is just one sessions
        else:
            for x in np.arange(0, 0.15, 0.001):
                axs.axvline(x, color='gray')
            offset = np.arange(0, len(rt), 1)
            axs.eventplot(rt, lineoffsets=offset, linewidth=7.5, colors=colors)
            axs.set_xlim([-0.2, 1.55])
            axs.set_ylabel('Trial #')
            axs.invert_yaxis()
            axs.set_xlabel('Response time (s)')
            axs.set_title(str(session))
            axs.legend(bbox_to_anchor=(0., 1.15, 1., .102), handles=patches, mode="expand", borderaxespad=0., ncol=1)
    
    # Prompt user for destination folder path or show the plot
    if save:
        fname = input('What path do you want to save the .jpg to?')
        fig.savefig(fname+mouse_data.id+'.jpg', bbox_inches='tight')   
    else:   
        plt.show()
    return


def plot_performance(mouse_data, average=False):
    ''' docstring
    '''
    # If you want to plot the average, check input and get n_sessions
    if average:
        title = 'Average'
        if not isinstance(mouse_data, list):
            raise TypeError('mouse_data should be a list of Mouse_Data Classes')
            return
        else:
            bigdict = {'mHit':[],'mMiss':[], 'mTotal':[], 'mRT':[],
                   'cHit':[],'cMiss':[], 'cTotal':[], 'cRT':[]}
            n_sessions = np.max([len(mouse.sessions) for mouse in mouse_data])
            xticks = np.arange(0, n_sessions, 1)
            massivedict = {n:bigdict.copy() for n in xticks}
    else:   
        title = mouse_data.id
        n_sessions = len(mouse_data.sessions)
        xticks = np.arange(0, n_sessions, 1)
        
    # Set plot 
    fig, axs = plt.subplots(4, figsize=(20, 20))
    fig.patch.set_facecolor('white')

    # Get the data to plot
    if average:
        for n_session in xticks:
            for mouse in mouse_data:
                session = mouse.sessions[n_session]
                testData = select_trialType(mouse.session_data[session], 'test')
                catchData = select_trialType(mouse.session_data[session], 'catch')

                # Gather data about mix (test) trials
                mHit, mMiss = get_hitnmiss(testData)
                mRT = np.mean(testData['response_t'])
                mRTsem = np.std(testData['response_t'].loc[testData['success']==True])
                mTotal = mHit + mMiss
                # Catch trials
                cHit, cMiss = get_hitnmiss(catchData)
                cRT = np.mean(catchData['response_t'])
                cRTsem = np.std(catchData['response_t'].loc[catchData['success']==True])
                cTotal = cHit + cMiss
                
                # Add to the big dict
                bigdata = {'mHit':mHit,'mMiss':mMiss, 'mTotal':mTotal, 'mRT':mRT,
                   'cHit':cHit,'cMiss':cMiss, 'cTotal':cTotal, 'cRT':cRT}
                [massivedict[n_session][key].append(bigdata[key]) for key in bigdata.keys()]

                # Unpack for ease of use after all sessions of a single mouse have been added
            mHit = np.mean(massivedict[n_session]['mHit'])
            mMiss = np.mean(massivedict[n_session]['mMiss'])
            mTotal = np.mean(massivedict[n_session]['mTotal'])
            mRT = np.mean(massivedict[n_session]['mRT'])
            cHit = np.mean(massivedict[n_session]['cHit'])
            cMiss = np.mean(massivedict[n_session]['cMiss'])
            cTotal = np.mean(massivedict[n_session]['cTotal'])
            cRT = np.mean(massivedict[n_session]['cRT'])
            
            n=n_session
            axs[0].bar([n-0.2, n+0.2], [mHit, cHit], color=['blue', 'red'], width=0.4)
            axs[1].bar([n-0.2, n+0.2], [(mHit/mTotal)*100, (cHit/cTotal)*100], color=['blue', 'red'], width=0.4)
            axs[2].bar([n-0.2, n+0.2], [(mHit/mTotal-cHit/cTotal)*100], color=['purple'], width=0.4)
            axs[3].bar([n-0.2, n+0.2], [mRT,cRT], color=['blue', 'red'], width=0.4)#, yerr=[mRTsem,cRTsem])

    else:
        for n, session in enumerate(mouse_data.sessions):
            # Gather data about mix (test) trials
            testData = select_trialType(mouse_data.session_data[session], 'test')
            mHit, mMiss = get_hitnmiss(testData)
            mRT = np.mean(testData['response_t'])
            mRTsem = np.std(testData['response_t'].loc[testData['success']==True])
            mTotal = mHit + mMiss
            # Catch trials
            catchData = select_trialType(mouse_data.session_data[session], 'catch')
            cHit, cMiss = get_hitnmiss(catchData)
            cRT = np.mean(catchData['response_t'])
            cRTsem = np.std(catchData['response_t'])
            cTotal = cHit + cMiss

            # Total hits and misses
            axs[0].bar([n-0.2, n+0.2], [mHit, cHit], color=['blue', 'red'], width=0.4)
            # Hit rate
            axs[1].bar([n-0.2, n+0.2], [(mHit/mTotal)*100, (cHit/cTotal)*100], color=['blue', 'red'], width=0.4)
            # Hits - Falsepositives
            axs[2].bar([n-0.2, n+0.2], [(mHit/mTotal-cHit/cTotal)*100], color=['purple'], width=0.4)
            # Reaction time
            axs[3].bar([n-0.2, n+0.2], [mRT,cRT], color=['blue', 'red'], width=0.4, yerr=[mRTsem/3,cRTsem/3])
            
    # Configure        
    fig.suptitle(title, y=.925)  
    axs[0].set_title('Microstim and Catch Hits')
    axs[0].set_ylim([0, 100])
    axs[0].set_xticks(xticks)
    axs[0].set_ylabel('Hits')    
    axs[1].set_ylim([0, 100])
    axs[1].set_xticks(xticks)
    axs[1].set_ylabel('Hit rate (%)')
    axs[1].axhline(y=[50], color='gray', linestyle='--')
    axs[2].set_ylim([0, 100])
    axs[2].set_xticks(xticks)
    axs[2].set_ylabel('Hits - FPs (%)')
    axs[3].set_ylim([0, 1.75])
    axs[3].set_ylabel('Reaction Time (s)')
    axs[3].set_xticks(xticks)
    axs[3].set_xlabel('Test day')
    # Add legend
    blue_patch = matplotlib.patches.Patch(color='blue', label='Microstim')
    red_patch = matplotlib.patches.Patch(color='red', label='Catch')
    legend = axs[0].legend(bbox_to_anchor=(0., 1.2, 1., .102), handles=[blue_patch, red_patch], loc='upper left', borderaxespad=0., ncol=1)
    
    plt.show()
    return


def plot_trialPerformance(mouse_data):
    '''docstring'''
    title = mouse_data.id
    n_sessions = len(mouse_data.sessions)
    xticks = np.arange(0, n_sessions, 1)
        
    # Set plot 
    fig, axs = plt.subplots(n_sessions,2, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    
    # Plot the progression of hit trials on the left side 
    # mixData corresponds to microstim trials
    for n, session in enumerate(mouse_data.sessions):
        mixData = select_trialType(mouse_data.session_data[session], 'test')
        mscore = []
        mhit = 0
        mtotal = len(mixData)
        for idx, mixTrial in mixData.iterrows():
            if mixTrial['success'] == True:
                mhit += 1
            mscore.append(mhit/mtotal*100)
        
        # Try to get a fit for mTrials
        x = np.arange(0, len(mscore), 1)
        y = mscore
        y_fit = fit_sigmoid(x, y)
        # axs[n,0].scatter(x, y, c='black')
        axs[n,0].plot(y_fit, c='blue')
        axs[n,0].set_ylabel(f'Session {n} \n Hits (%)')
        axs[n,0].set_ylim([0, 100])
        axs[n,0].grid()
        axs[-1,0].set_xlabel('Test Trials')

        
        catchData = select_trialType(mouse_data.session_data[session], 'catch')
        total = 0
        FP = 0
        score = []
        total = len(catchData)
        for idx, catchTrial in catchData.iterrows():
            if catchTrial['success'] == True:
                FP += 1
            score.append((FP/total)*100)
        
        # Try to get a fit
        x = np.arange(0, len(score), 1)
        y = score
        y_fit = fit_sigmoid(x, y)
        
        axs[n,1].plot(y_fit, c='red')
        axs[n,1].set_ylabel('False Positives (%)')
        axs[n,1].set_ylim([0, 30])
        axs[n,1].grid()
        axs[-1,1].set_xlabel('Catch Trials')

    fig.suptitle(title, y=.925)  
    plt.show()
    return


def plot_lickPerformance(mouse_data, save=False, peak=False):
    '''docstring'''
    # Check for peaking allowing the user to only see the plots of the first 2 sessions
    if peak:
        n_sessions = 2
    else:
        n_sessions = len(mouse_data.sessions)

    # Set figure basics 
    fig, axs = plt.subplots(1, n_sessions, figsize=(15, 10)) # Size plot according to the number of sessions
    plt.subplots_adjust(wspace=1.) 
    fig.patch.set_facecolor('white')
    fig.suptitle(str(mouse_data.id), y=1.05)

    # Create legend patches 
    gray_patch = matplotlib.patches.Patch(color='gray', label='Stimulus')

    # Get and plot data for every session
    for idx, session in enumerate(mouse_data.sessions):
        if peak and idx == n_sessions:
            break

        # Only stimulation trails are necessary to plot here
        stimTrials = select_trialType(mouse_data.session_data[session], trialType=1)

        # Get the licks, remove premature licks and zero on stim_t
        for i, trialData in enumerate(stimTrials.iterrows()):
            trialData = trialData[1] # Because the first value is the trialNumber, so slice
            curatedLicks = curateLicks(trialData)

            # Now add these licks to the rasterplot
            axs[idx].eventplot(curatedLicks, lineoffsets=i, colors='black', linewidths=0.75)
            [axs[idx].axvline(i, c='gray') for i in np.arange(0, 0.2, 0.01)]

        # Now invert y-axis for readability and customise plot
        axs[idx].invert_yaxis()
        axs[idx].set_xlim([-0.5, 1.7])
        axs[idx].set_ylabel('Trial #')
        axs[idx].set_xlabel('Time (s)')
        axs[idx].set_title(str(session))
        axs[idx].set_xticks(np.arange(0, 1.8, 0.5))

    # After all trials have been plotted adjust all plots
    plt.show()
    return


def get_hitnmiss(mouse_data):
    ''' docstring
    '''
    # Check input
    if not isinstance(mouse_data, pd.DataFrame):
        raise TypeError('mouse_data is not a DataFrame, please select a mouse_data.full_data or mouse_data.session_data[session]')
    
    # Select hits (correct trials) and misses (incorrect trials)
    hits = mouse_data.loc[mouse_data['success'] == True]
    misses = mouse_data.loc[mouse_data['success'] == False]
    return len(hits), len(misses)


def exponential(params, x, y):
    a, b = params
    residuals = y - (a * np.exp(b*x))
    return residuals


def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)


def fit_sigmoid(x, y):
    p0 = [max(y), np.median(x), 1, min(y)]
    popt, pcov = curve_fit(sigmoid, x,y,p0, maxfev=100000)
    L ,x0, k, b = popt
    y_fit = sigmoid(x, L, x0, k, b)
    return y_fit