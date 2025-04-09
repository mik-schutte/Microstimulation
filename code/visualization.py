''' visualization.py

    Contains functions needed for data visualization, such as plot_raster_rt.

    @mik-schutte
'''
import numpy as np
import pandas as pd
import matplotlib, os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helpers import *
from scipy.optimize import curve_fit 
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import deque
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({'font.size':16, 'font.family':'Arial', 'axes.facecolor':'white'})  


# Individual plots (no comparison)
def omni_plot(mouse):
    ''' The plot for showing all behavioural data: raster, licks, Plick, dprime
    '''
    fig = plt.figure(figsize=(15, 25))
    fig.suptitle(mouse.id, )
    fig.patch.set_facecolor('white')
    gs0 = GridSpec(4, 4, height_ratios=[1.5,1.5,1,1], wspace=0.3, hspace=0.3) # 

    # RASTER
    raster_ax = [fig.add_subplot(gs0[0, i]) for i in range(4)]
    # Create legend patches
    # gray_patch = matplotlib.patches.Patch(color='gray', label='Stimulus')
    orange_patch = matplotlib.patches.Patch(color='orange', label='False Positive')
    red_patch = matplotlib.patches.Patch(color='red', label='Miss')
    green_patch = matplotlib.patches.Patch(color='green', label='Hit')
    lightgreen_patch = matplotlib.patches.Patch(color='lightgreen', label='Correct Rejection')
    patches = [green_patch, red_patch, lightgreen_patch, orange_patch]

    # Get and plot data for every session
    for idx, session in enumerate(mouse.sessions):
        # For pairing only pair and mix data are important
        select_data = mouse.session_data[session].loc[(mouse.session_data[session]['trialType'] == 2)|(mouse.session_data[session]['trialType'] == 1)|(mouse.session_data[session]['trialType'] == 'pairData')]        
        
        # Aquire response time, xticks and datatype
        rt_full = select_data['response_t']
        x = np.arange(0, len(rt_full), 1) # Initiate x-raster_ax[idx]is for plotting
        rt = [[rt] for rt in rt_full] # plt.eventplot requires values as list to ascribe different offsets

        # Pick right color
        colors = []
        for _, trial in select_data.iterrows():
            trial_success = trial['success']
            if trial['trialType'] == 2:
                if trial_success:
                    c = 'orange'
                else:
                    c='lightgreen'
            elif trial['trialType'] == 1:
                if trial_success:
                    c = 'green'
                else:
                    c = 'red' 
            colors.append(c)
            
        # Now the plot
        # If there are multiple sessions
        for x in np.arange(0, 0.15, 0.001):
            raster_ax[idx].axvline(x, color='gray')
        offset = np.arange(0, len(rt), 1)
        raster_ax[idx].eventplot(rt, lineoffsets=offset, linewidth=7.5, colors=colors)
        raster_ax[idx].set_xlim([-0.2, 1.71])
        raster_ax[idx].set_ylabel('Trial #')
        raster_ax[idx].invert_yaxis()
        raster_ax[idx].set_ylim([len(rt), 0])
        raster_ax[idx].set_xlabel('Response time (s)')
        raster_ax[idx].set_title(str(session))
        raster_ax[idx].set_xticks(np.arange(0, 1.55, 0.5))
        raster_ax[0].legend(bbox_to_anchor=(0., 1.15, 1., .02), handles=patches, mode="expand", borderaxespad=0., ncol=1, frameon=False)

    # LICKING
    lick_ax = [fig.add_subplot(gs0[1, i]) for i in range(4)]
    # Plot lick performance for every session
    for idx, session in enumerate(mouse.sessions):
        stim_trials = select_trialType(mouse.session_data[session], trialType=1)

        # Plot the licks
        for i, trialData in enumerate(stim_trials.iterrows()):
            trialData = trialData[1]
            curatedLicks = curateLicks(trialData)
            lick_ax[idx].eventplot(curatedLicks, lineoffsets=i, colors='black', linewidths=0.75)
            [lick_ax[idx].axvline(i, c='gray') for i in np.arange(0, 0.2, 0.01)]

        # Format
        lick_ax[idx].invert_yaxis()
        lick_ax[idx].set_xlim([-0.5, 1.7])
        lick_ax[idx].set_ylabel('Stim Trial #')
        lick_ax[idx].set_xlabel('Time (s)')
        lick_ax[idx].set_title(str(session))
        lick_ax[idx].set_xticks(np.arange(-0.5, 1.8, 0.5))

    # PLICK
    Plick_ax = fig.add_subplot(gs0[2, 0:2])
    # Get microstimulation (mstimP) and catch trial (catchP) lick probabilities
    mstimP, catchP = get_PLick(mouse)

    # Create a new figure and axis for plotting

    # Plot microstimulation lick probability ('mstimP') in blueviolet color and ('catchP') in gray color
    Plick_ax.plot(mstimP, color='blueviolet', alpha=1, label=r'$\mu$Stim')
    Plick_ax.plot(catchP, color='gray', alpha=1, label='Catch') 
    x = np.arange(0, len(mstimP), 1)
    Plick_ax.scatter(x, mstimP, c='blueviolet')  # Scatter plot for 'stim' trials
    Plick_ax.scatter(x, catchP, c='gray')  # Scatter plot for 'catch' trials

    # Set plot title and labels
    Plick_ax.set_title(mouse.id)  # Set the plot title to mouse id (if desired)
    Plick_ax.set_xticks([0,1,2,3])
    Plick_ax.set_xticklabels([1, 2, 3, 4])  # Set custom x-axis tick labels
    Plick_ax.set_xlabel('Session')  # Set x-axis label to 'Session'
    Plick_ax.set_ylabel('P(lick)')  # Set y-axis label to 'P(lick)'

    # Set y-axis limits between -0.05 and 1.05 for a proper range of probabilities
    Plick_ax.set_ylim([-0.05, 1.05])
    Plick_ax.legend(loc='upper left')

    # RT
    RT_ax = fig.add_subplot(gs0[2, 2:4])
    # Initialize dictionary to store response times
    RT_dict = {'stim': [], 'catch': []}

    # Iterate over each mouse in the list
    rt_stim_indi = []
    rt_catch_indi = []

    # Iterate over each session of the mouse
    for session_name in mouse.sessions:
        session = mouse.session_data[session_name]

        # Calculate average response time for 'stim' trials
        stim_trials = select_trialType(session, 'test')
        stim_trials = stim_trials.loc[stim_trials['success'] == True]  # Filter successful trials
        rt_stim = np.average(stim_trials['response_t'])
        rt_stim_indi.append(rt_stim)

        # Calculate average response time for 'catch' trials
        catch_trials = select_trialType(session, 'catch')
        catch_trials = catch_trials.loc[catch_trials['success'] == True]  # Filter successful trials
        rt_catch = np.average(catch_trials['response_t'])
        rt_catch_indi.append(rt_catch)

        # Append response times for the current mouse to RT_dict
        RT_dict['stim'].append(rt_stim_indi)
        RT_dict['catch'].append(rt_catch_indi)

    # Calculate average and standard deviation across mice for each session
    stim_avg = np.average(RT_dict['stim'], axis=0)
    catch_avg = np.average(RT_dict['catch'], axis=0)
    stim_std = np.std(RT_dict['stim'], axis=1)
    catch_std = np.std(RT_dict['catch'], axis=1)

    # Plot average response times and add scatter points
    x = np.arange(len(stim_avg))
    RT_ax.plot(stim_avg, label='Stim', c='blueviolet')
    RT_ax.scatter(x, stim_avg, c='blueviolet')  # Scatter plot for 'stim' trials
    RT_ax.plot(catch_avg, label='Catch', c='gray')
    RT_ax.scatter(x, catch_avg, c='gray')  # Scatter plot for 'catch' trials

    # If multiple mice are plotted, show error bars representing standard deviation
    RT_ax.errorbar(x, stim_avg, yerr=stim_std, c='blueviolet', capsize=5)
    RT_ax.errorbar(x, catch_avg, yerr=catch_std, c='gray', capsize=5)

    # Set plot limits, ticks, labels, and display the legend
    RT_ax.set_ylim([0, 1.2])
    RT_ax.set_yticks(np.arange(0, 1.2, 0.2))
    RT_ax.set_xlim([-0.5, len(stim_avg) - 0.5])  # Adjust x-axis limits based on number of sessions
    RT_ax.set_xticks(x)
    RT_ax.set_xticklabels([1,2,3,4])
    RT_ax.set_ylabel('Response time (s)')
    RT_ax.set_xlabel('Session')

    # Rate correct
    rate_ax = fig.add_subplot(gs0[3, 0:2])
    n_sessions = len(mouse.sessions)
    xticks = np.arange(0, n_sessions, 1)


    # Get the data to plot
    for n_session in xticks:
        session = mouse.sessions[n_session]
        testData = select_trialType(mouse.session_data[session], 'test')
        catchData = select_trialType(mouse.session_data[session], 'catch')

        # Gather data about microstim and catch trials
        mHit, mMiss = get_hitnmiss(testData)
        mTotal = mHit + mMiss
        cHit, cMiss = get_hitnmiss(catchData)
        cTotal = cHit + cMiss

        # For every individual animal during each session calculate the rate
        hit_rate = mHit / mTotal *100
        FP_rate = cHit / cTotal * 100
        effect_rate = hit_rate - FP_rate
        rate_ax.bar([n_session-0.2, n_session, n_session+0.2], [hit_rate, FP_rate, effect_rate], color=['blueviolet', 'gray', 'plum'], width=0.2)
        
                
        # Formatting
        # fig.suptitle(title, y=.925)  
        rate_ax.set_ylabel('Percentage (%)')
        rate_ax.set_ylim([0, 110])

        rate_ax.set_xticks(xticks)
        rate_ax.set_xticklabels([1,2,3,4])
        rate_ax.set_xlabel('Session')
        
        # Add legend
        blue_patch = matplotlib.patches.Patch(color='blueviolet', label='Hit')
        red_patch = matplotlib.patches.Patch(color='gray', label='FP')
        plum_patch = matplotlib.patches.Patch(color='plum', label='Hit - FPs')
        rate_ax.legend(bbox_to_anchor=(0.005, 0.89, 1., .102), handles=[blue_patch, red_patch, plum_patch], loc='upper left', borderaxespad=0., ncol=1)

    d_ax = fig.add_subplot(gs0[3, 2:4])
    mega_d = []
    d_prime_list = calc_d_prime(mouse)
    mega_d.append(d_prime_list)

    avg_list, std_list = get_avg_std_threshold(mega_d, max_sessions=4)

    # Ploterdeplot
    # Individual lines and points
    [d_ax.plot(d_prime, c='black', alpha=0.25) for d_prime in mega_d] 
    [d_ax.scatter(x=[0,1,2,3],y=d_prime, c='black', alpha=0.3) for d_prime in mega_d]  
    # Average
    d_ax.plot(avg_list, c='black', linewidth=2)
    d_ax.scatter(x=[0,1,2,3], y=avg_list, c='black', linewidths=2)

    # Format
    d_ax.set_ylim([-0.05,5])
    d_ax.set_ylabel('d\' (Sensitivity Index)')
    d_ax.set_xticks([0,1,2,3], [1,2,3,4])
    d_ax.set_xlabel('Session')
    plt.show()

matplotlib.rcParams.update({'font.size':16, 'font.family':'Arial', 'axes.facecolor':'white'})   

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
    fig, axs = plt.subplots(1, n_sessions, figsize=(12, 6)) # Size plot according to the number of sessions
    plt.subplots_adjust(wspace=1.) 
    fig.patch.set_facecolor('white')
    fig.suptitle(str(mouse_data.id), y=1.0)
    
    # Create legend patches
    # gray_patch = matplotlib.patches.Patch(color='gray', label='Stimulus')
    orange_patch = matplotlib.patches.Patch(color='orange', label='False Positive')
    red_patch = matplotlib.patches.Patch(color='red', label='Miss')
    green_patch = matplotlib.patches.Patch(color='green', label='Hit')
    lightgreen_patch = matplotlib.patches.Patch(color='lightgreen', label='Correct Rejection')
    
    # Get and plot data for every session
    for idx, session in enumerate(mouse_data.sessions):
        if peak and idx == n_sessions:
            break
        colors = []
        
        # For pairing only pair and mix data are important
        # if catch:
        select_data = mouse_data.session_data[session].loc[(mouse_data.session_data[session]['trialType'] == 2)|(mouse_data.session_data[session]['trialType'] == 1)|(mouse_data.session_data[session]['trialType'] == 'pairData')]        
        patches = [green_patch, red_patch, lightgreen_patch, orange_patch]
        
        # Aquire response time, xticks and datatype
        rt_full = select_data['response_t']
        x = np.arange(0, len(rt_full), 1) # Initiate x-axis for plotting
        rt = [[rt] for rt in rt_full] # plt.eventplot requires values as list to ascribe different offsets
        # dtype = [[dtype] for dtype in select_data['trialType']]

        # Pick right color
        for _, trial in select_data.iterrows():
            trial_success = trial['success']
            if trial['trialType'] == 2:
                if trial_success:
                    c = 'orange'
                else:
                    c='lightgreen'
            elif trial['trialType'] == 1:
                if trial_success:
                    c = 'green'
                else:
                    c = 'red'
                    
            colors.append(c)
            
        # Now the plot
        # If there are multiple sessions
        if len(mouse_data.sessions) > 1:
            for x in np.arange(0, 0.15, 0.001):
                axs[idx].axvline(x, color='gray')
            offset = np.arange(0, len(rt), 1)
            axs[idx].eventplot(rt, lineoffsets=offset, linewidth=7.5, colors=colors)
            axs[idx].set_xlim([-0.2, 1.71])
            axs[idx].set_ylabel('Trial #')
            axs[idx].invert_yaxis()
            axs[idx].set_ylim([150, 0])
            axs[idx].set_xlabel('Response time (s)')
            axs[idx].set_title(str(session))
            axs[idx].set_xticks(np.arange(0, 1.55, 0.5))
            axs[0].legend(bbox_to_anchor=(0., 1.15, 1., .02), handles=patches, mode="expand", borderaxespad=0., ncol=1, frameon=False)
        
        # If there is just one sessions
        else:
            for x in np.arange(0, 0.15, 0.001):
                axs.axvline(x, color='gray')
            offset = np.arange(0, len(rt), 1)
            axs.eventplot(rt, lineoffsets=offset, linewidth=7.5, colors=colors)
            axs.set_xlim([-0.2, 1.71])
            axs.set_ylabel('Trial #')
            axs.invert_yaxis()
            axs.set_xlabel('Response time (s)')
            axs.set_title(str(session))
            axs.legend(bbox_to_anchor=(0., 1.15, 1., .102), handles=patches, mode="expand", borderaxespad=0., ncol=1, frameon=False)
    
    # Prompt user for destination folder path or show the plot
    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save }')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600) 
    else:   
        plt.show()
    return


def plot_performance(mouse_data, save=False, average=False):
    ''' Barplot showing the percentage hits, false positives and hits-FP for each session of an individual animal

    Parameters:
        mouse_data (list or Mouse_Data): A (list of) Mouse_Data instances.
        average (bool, optional): If True, plot average performance over sessions.

    Returns:
        matplotlib.pyplot 
    '''
    # Check input and initialize data structures for averaging
    if average:
        if not isinstance(mouse_data, list):
            raise TypeError('mouse_data should be a list of Mouse_Data Classes')
        # Determine number of sessions and prepare x-axis ticks
        bigdict = {'hit_rate':[], 'FP_rate':[], 'effect_rate':[]}
        n_sessions = np.max([len(mouse.sessions) for mouse in mouse_data])
        xticks = np.arange(0, n_sessions, 1)
        massivedict = {n:bigdict.copy() for n in xticks}
        title = 'Average'
    # Prepare data for individual mouse plot
    else:   
        title = mouse_data.id
        n_sessions = len(mouse_data.sessions)
        xticks = np.arange(0, n_sessions, 1)
        
    # Set plot 
    fig, axs = plt.subplots(1, figsize=(8, 4))
    fig.patch.set_facecolor('white')

    # Get the data to plot
    if average:
        # For each session get the rate for each animal
        for n_session in xticks:
            bigdict = {'hit_rate':[], 'FP_rate':[], 'effect_rate':[]}
            for mouse in mouse_data:
                session = mouse.sessions[n_session]
                testData = select_trialType(mouse.session_data[session], 'test')
                catchData = select_trialType(mouse.session_data[session], 'catch')

                # Gather data about microstim and catch trials
                mHit, mMiss = get_hitnmiss(testData)
                mTotal = mHit + mMiss
                cHit, cMiss = get_hitnmiss(catchData)
                cTotal = cHit + cMiss

                # For every individual animal during each session calculate the rate and add to the big dict
                hit_rate = mHit / mTotal *100
                FP_rate = cHit / cTotal * 100
                effect_rate = hit_rate - FP_rate
                bigdict['hit_rate'].append(hit_rate)
                bigdict['FP_rate'].append(FP_rate)
                bigdict['effect_rate'].append(effect_rate)

                # Plot individual points
                plt.scatter(x=n_session-0.2, y=hit_rate, color='green', zorder=10, edgecolors='black', linewidths=0.5, s=10)
                plt.scatter(x=n_session, y=FP_rate, color='orange', zorder=10, edgecolors='black', linewidths=0.5, s=10)
                plt.scatter(x=n_session+0.2, y=effect_rate, color='blue', zorder=10, edgecolors='black', linewidths=0.5, s=10)
            
            # Store bigdict in massivedict
            massivedict[n_session] = bigdict

            # Plot
            # Get the average rates for each session and plot them as a bar.  
            hit_rate = np.mean(massivedict[n_session]['hit_rate']) 
            FP_rate = np.mean(massivedict[n_session]['FP_rate'])
            effect_rate = np.mean(massivedict[n_session]['effect_rate'])
            hit_std = np.std(massivedict[n_session]['hit_rate'])/len(mouse_data)
            FP_std = np.std(massivedict[n_session]['FP_rate'])/len(mouse_data)
            effect_std = np.std(massivedict[n_session]['effect_rate'])/len(mouse_data)
            axs.bar([n_session-0.2, n_session, n_session+0.2], [hit_rate, FP_rate, effect_rate], color=['green', 'orange', 'blue'], width=0.2, yerr=[hit_std, FP_std, effect_std], capsize=2)

    # For plotting the data of an individual animal
    else:
        for n_session in xticks:
            mouse = mouse_data
            session = mouse.sessions[n_session]
            testData = select_trialType(mouse.session_data[session], 'test')
            catchData = select_trialType(mouse.session_data[session], 'catch')

            # Gather data about microstim and catch trials
            mHit, mMiss = get_hitnmiss(testData)
            mTotal = mHit + mMiss
            cHit, cMiss = get_hitnmiss(catchData)
            cTotal = cHit + cMiss

            # For every individual animal during each session calculate the rate
            hit_rate = mHit / mTotal *100
            FP_rate = cHit / cTotal * 100
            effect_rate = hit_rate - FP_rate
            axs.bar([n_session-0.2, n_session, n_session+0.2], [hit_rate, FP_rate, effect_rate], color=['green', 'orange', 'blue'], width=0.2)
            
            
    # Formatting
    fig.suptitle(title, y=.95)  
    axs.set_ylabel('Percentage (%)')
    axs.set_ylim([0, 110])

    axs.set_xticks(xticks)
    # axs.set_xticklabels([1,2,3,4]) # When enabled it can yield error if you have too many sessions
    axs.set_xlabel('Session')
    
    # Add legend
    blue_patch = matplotlib.patches.Patch(color='green', label='Hit')
    red_patch = matplotlib.patches.Patch(color='orange', label='FP')
    plum_patch = matplotlib.patches.Patch(color='blue', label='Hit - FPs')
    axs.legend(bbox_to_anchor=(0.005, 0.89, 1., .102), handles=[blue_patch, red_patch, plum_patch], loc='upper left', borderaxespad=0., ncol=1)

    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save }')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600) 
    
    plt.show()
    return


def plot_d_prime(mouse_data, save=False):
    if not isinstance(mouse_data, list):
        mouse_data = [mouse_data]
    mega_d = []
    for mouse in mouse_data:
        d_prime_list = calc_d_prime(mouse)
        mega_d.append(d_prime_list)

    avg_list, std_list = get_avg_std_threshold(mega_d, max_sessions=4)

    # Ploterdeplot
    fig = plt.figure(figsize=(3,6))
    # Individual lines and points
    [plt.plot(d_prime, c='black', alpha=0.25) for d_prime in mega_d] 
    [plt.scatter(x=[0,1,2,3],y=d_prime, c='black', alpha=0.3) for d_prime in mega_d]  
    # Average
    plt.plot(avg_list, c='black', linewidth=2)
    plt.scatter(x=[0,1,2,3], y=avg_list, c='black', linewidths=2)

    # SEM
    sem_list = np.array(std_list)/len(mouse_data)
    if len(mouse_data) > 1:
        plt.errorbar([0,1,2,3], avg_list, yerr=sem_list, c='black',capsize=5)

    # Format
    # plt.ylim([-0.05,4])
    # plt.yticks([0,1,2,3,4])
    plt.ylabel('d\' (Sensitivity Index)')
    plt.xticks([0,1,2,3], [1,2,3,4])
    plt.xlabel('Session')

    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save }')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600) 
    plt.show()


def plot_lickPerformance(mouse_data, save=False, show=True, plotCatch=True):
    # TODO why are some of the first licks occluded by the stimulus?
    """Plots lick performance for stimulus and catch trials in a 2-row raster plot"""

    # Check for peaking (limit to first 2 sessions if peak=True)
    n_sessions = len(mouse_data.sessions)

    # Set figure basics 
    fig, axs = plt.subplots(2, n_sessions, figsize=(15, 12), sharex=False) # 2 rows: Stimuli (top) & Catch (bottom)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spacing
    fig.patch.set_facecolor('white')
    fig.suptitle(str(mouse_data.id), y=0.955)

    # Get and plot data for every session
    for idx, session in enumerate(mouse_data.sessions):

        # Select Stimulus and Catch trials
        stimTrials = select_trialType(mouse_data.session_data[session], trialType=1)
        catchTrials = select_trialType(mouse_data.session_data[session], trialType=2)

        # ---- PLOT STIMULUS TRIALS (Top Row: axs[0, idx]) ----
        for i, trialData in enumerate(stimTrials.iterrows()):
            trialData = trialData[1]  # Extract trial data
            curatedLicks = curateLicks(trialData)
            if len(curatedLicks) > 0:
                axs[0, idx].eventplot(curatedLicks[1:], lineoffsets=i, colors='black', linewidths=0.75, zorder=1)
                axs[0, idx].eventplot([curatedLicks[0]], lineoffsets=i, colors='green', linewidths=5, zorder=2)

        axs[0, idx].invert_yaxis()
        axs[0, idx].set_xlim([-0.5, 1.7])
        axs[0, idx].set_xticks([-0.5, 0, 0.5, 1, 1.5])
        axs[0, idx].set_ylim([len(stimTrials)+1, 0])
        axs[0, idx].set_ylabel(r'$\mu$Stim trials')
        axs[0, idx].set_xlabel('Time (s)')
        axs[0, idx].set_title(f'{session}')
        [axs[0, idx].axvline(i, c='gray') for i in np.arange(0, 0.2, 0.01)] # Mark stimulus 

        # ---- PLOT CATCH TRIALS (Bottom Row: axs[1, idx]) ----
        for i, trialData in enumerate(catchTrials.iterrows()):
            trialData = trialData[1]  # Extract trial data
            curatedLicks = curateLicks(trialData)
            if len(curatedLicks) > 1:
                axs[1, idx].eventplot(curatedLicks, lineoffsets=i, colors='black', linewidths=0.75)  
                axs[1, idx].eventplot([curatedLicks[0]], lineoffsets=i, colors='orange', linewidths=3, zorder=2)

        axs[1, idx].invert_yaxis()
        axs[1, idx].set_xlim([-0.5, 1.7])
        axs[1, idx].set_xticks([-0.5, 0, 0.5, 1, 1.5])
        axs[1, idx].set_ylim([len(catchTrials)+1, 0])
        axs[1, idx].set_ylabel('Catch Trial #')
        axs[1, idx].set_xlabel('Time (s)')
        axs[1, idx].set_title(f'{session}')
        [axs[1, idx].axvline(i, c='gray') for i in np.arange(0, 0.2, 0.01)] # Mark stimulus 

    # Save figure if requested
    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save}')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600)

    # # For when you only want the axs to use as a subplot
    # if show:
    #     plt.show() # TODO this doesn't yield a nice plot if you only want stim trials
    # if plotCatch:
    #     return axs
    # else: # If plotCatch is False then only return the row with stimulus trial licks (Hits)
    #     return axs[0,:]
    
    plt.show
    return


def plot_FLicks(mouse, save=False):
    """
    Plot the kernel density estimation (KDE) of first lick response times 
    for mStim and Catch trials from the last session of each mouse.

    Parameters:
    - mice: A single mouse object or a list of mouse objects
    """

    RTs_mstim = []
    RTs_catch = []

    # Select the last session
    session = mouse.sessions[-1]  # Last session
    session_data = mouse.session_data[session]

    # Select trial types and filter for success
    mstim = select_trialType(session_data, 'test')
    catch = select_trialType(session_data, 'catch')
    mstim = mstim[mstim['success'] == True]
    catch = catch[catch['success'] == True]

    if not mstim.empty:
        RTs_mstim.append(mstim['response_t'])

    if not catch.empty:
        RTs_catch.append(catch['response_t'])

    # Combine into single series (skip if no data)
    RTs_mstim = pd.concat(RTs_mstim) if RTs_mstim else pd.Series(dtype=float)
    RTs_catch = pd.concat(RTs_catch) if RTs_catch else pd.Series(dtype=float)

    # X-axis for KDE
    x = np.arange(-1, 2, 0.01)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    [ax.axvline(i, ymin=0, ymax=3, c='gray', alpha=0.1) for i in np.arange(0, 0.2, 0.001)]

    # KDE lines
    if not RTs_mstim.empty:
        density_mstim = gaussian_kde(RTs_mstim)
        ax.plot(x, density_mstim(x), c='green', label=r'$\mu$Stim')

    if not RTs_catch.empty:
        density_catch = gaussian_kde(RTs_catch)
        ax.plot(x, density_catch(x), c='gray', label='Catch')
    else:
        ax.plot(x, np.zeros(len(x)), c='gray', label='Catch (no data)')

    # Formatting
    ax.set_xlim([-1, 2])
    ax.set_ylim([-0.2, 5])
    ax.set_xlabel('Time from Stimulus Onset (s)')
    ax.set_ylabel('First Licks (Hz)')
    ax.set_title('First Licks (KDE) — Last Session')
    ax.legend()
    plt.tight_layout()

    # Save the plot
    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save }')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600) 
    plt.show()


def plot_Plick(mouse_data, save=False):
    """
    Plot lick probability (Plick) for microstimulation ('mstimP') and catch trials ('catchP').

    Parameters:
        mouse (Mouse_Data): Mouse_Data instance containing lick probability data.

    Returns:
        None
    """
    # Get microstimulation (mstimP) and catch trial (catchP) lick probabilities
    mstimP, catchP = get_PLick(mouse_data)

    # Create a new figure and axis for plotting
    fig, axs = plt.subplots(figsize=(6, 8))

    # Plot microstimulation lick probability ('mstimP') in green color and ('catchP') in orange color
    axs.plot(mstimP, color='green', alpha=1, label=r'$\mu$Stim')
    axs.plot(catchP, color='orange', alpha=1, label='Catch') 
    x = np.arange(0, len(mstimP), 1)
    axs.scatter(x, mstimP, c='green')  # Scatter plot for 'stim' trials
    axs.scatter(x, catchP, c='orange')  # Scatter plot for 'catch' trials

    # Set plot title and labels
    axs.set_title(mouse_data.id)  # Set the plot title to mouse id (if desired)

    xticks = np.arange(0, len(mstimP), 1)
    axs.set_xticks(xticks)
    axs.set_xticklabels(xticks+1)  # Set custom x-axis tick labels
    axs.set_xlabel('Session')  # Set x-axis label to 'Session'
    axs.set_ylabel('P(lick)')  # Set y-axis label to 'P(lick)'

    # Set y-axis limits between -0.05 and 1.05 for a proper range of probabilities
    axs.set_ylim([-0.05, 1.05])

    # Display legend for the plot
    axs.legend(loc='upper left')
    # Save figure if requested
    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save}')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600)

    # Show the plot
    plt.show()
    return

def plot_metrics(mouse, save=False):
    ''' The plot for showing all behavioural data: Plick, dprime, Rate Correct, RT
    '''
    fig = plt.figure(figsize=(14, 7))  # Adjusted figure size for 2 rows x 2 columns
    fig.suptitle(mouse.id)
    fig.patch.set_facecolor('white')
    gs0 = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.25)  # Adjusted for 2 rows, 2 columns
    
    # PLICK
    Plick_ax = fig.add_subplot(gs0[0, 0])
    mstimP, catchP = get_PLick(mouse)
    Plick_ax.plot(mstimP, color='green', alpha=1, label=r'$\mu$Stim')
    Plick_ax.plot(catchP, color='orange', alpha=1, label='Catch') 
    xticks = np.arange(0, len(mstimP), 1)
    Plick_ax.scatter(xticks, mstimP, c='green')
    Plick_ax.scatter(xticks, catchP, c='orange')
    Plick_ax.set_xticks(xticks)
    Plick_ax.set_xticklabels(xticks+1)
    Plick_ax.set_xlabel('Session')
    Plick_ax.set_ylabel('P(lick)')
    Plick_ax.set_ylim([-0.05, 1.05])
    Plick_ax.legend(loc='upper left')

    # RT
    RT_ax = fig.add_subplot(gs0[0, 1])
    RT_dict = {'stim': [], 'catch': []}
    rt_stim_indi = []
    rt_catch_indi = []

    for session_name in mouse.sessions:
        session = mouse.session_data[session_name]

        # HIT
        stim_trials = select_trialType(session, 'test')
        stim_trials = stim_trials.loc[stim_trials['success'] == True] # Because only trials that had succesful lick have a meaningful response time
        if len(stim_trials) > 0:
            rt_stim = np.average(stim_trials['response_t'])
        else:
            rt_stim = 0
        rt_stim_indi.append(rt_stim)

     #FALSE POSITIVE
        catch_trials = select_trialType(session, 'catch')
        catch_trials = catch_trials.loc[catch_trials['success'] == True]
        if len(catch_trials) > 0:
            rt_catch = np.average(catch_trials['response_t'])
        else:
            rt_catch = 0
        rt_catch_indi.append(rt_catch)

        RT_dict['stim'].append(rt_stim_indi)
        RT_dict['catch'].append(rt_catch_indi)

    stim_avg = np.average(RT_dict['stim'], axis=0)
    catch_avg = np.average(RT_dict['catch'], axis=0)
    stim_std = np.std(RT_dict['stim'], axis=1)
    catch_std = np.std(RT_dict['catch'], axis=1)
    x = np.arange(len(stim_avg))
    RT_ax.plot(stim_avg, label='Stim', c='green')
    RT_ax.scatter(x, stim_avg, c='green')
    RT_ax.plot(catch_avg, label='Catch', c='orange')
    RT_ax.scatter(x, catch_avg, c='orange')
    RT_ax.errorbar(x, stim_avg, yerr=stim_std, c='green', capsize=5)
    RT_ax.errorbar(x, catch_avg, yerr=catch_std, c='orange', capsize=5)
    RT_ax.set_ylim([0, 1.7])
    # RT_ax.set_yticks(np.arange(0, 1.5, 0.2))
    RT_ax.set_xlim([-0.5, len(stim_avg) - 0.5])
    RT_ax.set_xticks(x)
    RT_ax.set_xticklabels(x+1)
    RT_ax.set_ylabel('Response time (s)')
    RT_ax.set_xlabel('Session')

    # Rate correct
    rate_ax = fig.add_subplot(gs0[1, 0])
    n_sessions = len(mouse.sessions)
    xticks = np.arange(0, n_sessions, 1)

    for n_session in xticks:
        session = mouse.sessions[n_session]
        testData = select_trialType(mouse.session_data[session], 'test')
        catchData = select_trialType(mouse.session_data[session], 'catch')
        mHit, mMiss = get_hitnmiss(testData)
        mTotal = mHit + mMiss
        cHit, cMiss = get_hitnmiss(catchData)
        cTotal = cHit + cMiss
        hit_rate = mHit / mTotal * 100
        FP_rate = cHit / cTotal * 100
        effect_rate = hit_rate - FP_rate
        rate_ax.bar([n_session - 0.2, n_session, n_session + 0.2], [hit_rate, FP_rate, effect_rate], color=['green', 'orange', 'black'], width=0.2)

    rate_ax.set_ylabel('Percentage (%)')
    rate_ax.set_ylim([0, 110])
    rate_ax.set_xticks(xticks)
    rate_ax.set_xticklabels(xticks + 1)
    rate_ax.set_xlabel('Session')
    blue_patch = matplotlib.patches.Patch(color='green', label='Hit')
    red_patch = matplotlib.patches.Patch(color='orange', label='FP')
    yellow_patch = matplotlib.patches.Patch(color='black', label='Hit - FPs')
    rate_ax.legend(bbox_to_anchor=(0.005, 0.89, 1., .102), handles=[blue_patch, red_patch, yellow_patch], loc='upper left', borderaxespad=0., ncol=1)

    # d-prime
    d_ax = fig.add_subplot(gs0[1, 1])
    mega_d = []
    d_prime_list = calc_d_prime(mouse)
    mega_d.append(d_prime_list)
    avg_list, std_list = get_avg_std_threshold(mega_d, max_sessions=len(mouse.sessions))
    [d_ax.plot(d_prime, c='black', alpha=0.25) for d_prime in mega_d]
    [d_ax.scatter(x=np.arange(0, len(d_prime), 1), y=d_prime, c='black', alpha=0.3) for d_prime in mega_d]
    d_ax.plot(avg_list, c='black', linewidth=2)
    d_ax.scatter(x=xticks, y=avg_list, c='black', linewidths=2)
    d_ax.set_ylim([-0.05, 5])
    d_ax.set_ylabel('d\' (Sensitivity Index)')
    d_ax.set_xticks(xticks)
    d_ax.set_xticklabels(xticks + 1)
    d_ax.set_xlabel('Session')

    # Adjust spacing
    fig.subplots_adjust(top=0.92)  # Reduce the space between the title and the plots

    # Save figure if requested
    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save}')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600)

    plt.show()
    return


def plot_RT(mouse_data, save=False, sem=True):
    """
    Plot response time (RT) analysis for a list of mouse data sessions.

    This function calculates and visualizes the average response times for 'stim' and 'catch' trials across
    multiple sessions of one or more mice.

    Parameters:
        mouse_data (Mouse_Data or list): A Mouse_Data instance or a list of Mouse_Data instances.

    Returns:
        None
    """
    # Initialize dictionary to store response times
    RT_dict = {'stim': [], 'catch': []}

    # For each animal create a new list that will contain lists of all response times of the trials for each sessio
    rt_stim_indi = []
    rt_catch_indi = []

    # Iterate over each session of the mouse
    for session_name in mouse_data.sessions:
        session_data = mouse_data.session_data[session_name]

        # Calculate average response time for 'stim' trials
        stim_rts = get_RT(session_data, trialType=1)
        rt_stim_indi.append(stim_rts.values)

        # Calculate average response time for 'catch' trials
        catch_rts = get_RT(session_data, trialType=2)
        rt_catch_indi.append(catch_rts.values)

    # Calculate statistics
    stim_mean = [np.mean(rt_stim_session) for rt_stim_session in rt_stim_indi if len(rt_stim_session) > 0]
    stim_std = [np.std(rt_stim_session) for rt_stim_session in rt_stim_indi if len(rt_stim_session) > 0]
    stim_sem = [np.std(rt_stim_session)/np.sqrt(len(rt_stim_session)) for rt_stim_session in rt_stim_indi if len(rt_stim_session) > 0]
    # Catch
    catch_mean = [np.mean(rt_catch_session) for rt_catch_session in rt_catch_indi if len(rt_catch_session) > 0]
    catch_std = [np.std(rt_catch_session) for rt_catch_session in rt_catch_indi if len(rt_catch_session) > 0]
    catch_sem = [np.std(rt_catch_session)/np.sqrt(len(rt_catch_session)) for rt_catch_session in rt_catch_indi if len(rt_catch_session) > 0]

    # Prepare x-axis values (sessions)
    x_stim = np.arange(len(stim_mean))
    x_catch = np.arange(len(catch_mean))
    # # Create a new figure and axis for plotting
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot average response times and add scatter points
    ax.plot(stim_mean, label='Stim', c='green', zorder=2)
    ax.scatter(x_stim, stim_mean, c='green', zorder=2)  # Scatter plot for 'stim' trials
    ax.plot(catch_mean, label='Catch', c='orange', zorder=2)
    ax.scatter(x_catch, catch_mean, c='orange', zorder=2)  # Scatter plot for 'catch' trials

    # If multiple mice are plotted, show error bars representing standard deviation
    if sem:
        ax.errorbar(x_stim, stim_mean, yerr=stim_sem, c='black', capsize=4, zorder=1)
        ax.errorbar(x_catch, catch_mean, yerr=catch_sem, c='black', capsize=4, zorder=1)
    else:
        ax.errorbar(x_stim, stim_mean, yerr=stim_std, c='black', capsize=4, zorder=1)
        ax.errorbar(x_catch, catch_mean, yerr=catch_std, c='black', capsize=4, zorder=1)

    # Set plot limits, ticks, labels, and display the legend
    ax.set_ylim([0, 1.6])
    ax.set_yticks(np.arange(0.0, 1.8, 0.2))

    # ax.set_xlim([-0.5, len(stim_mean) - 0.5])  # Adjust x-axis limits based on number of sessions
    xticks = np.arange(0, len(stim_mean), 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks+1)
    ax.set_ylabel('Response time (s)')
    ax.set_xlabel('Session')
    ax.legend(loc='upper right')

    # Save figure if requested
    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save}')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600)

    # Display the plot
    plt.show()
    return


def plot_d_prime(mouse_data, save=False):
    if not isinstance(mouse_data, list):
        mouse_data = [mouse_data]
    mega_d = [] # Create a list of dprimes for each session
    for mouse in mouse_data:
        d_prime_list = calc_d_prime(mouse)
        mega_d.append(d_prime_list)

    # TODO why individual function and max_sessions given?
    # TODO remove this function and replace by np.mean and np.std
    # TODO what is the std from, dprime over single session? std is zero if iterating over single session
    avg_list, std_list = get_avg_std_threshold(mega_d, max_sessions=len(mouse.sessions)) # avg_list is the average dprime of each session in an individual mouse

    # Ploterdeplot
    fig = plt.figure(figsize=(3,6))
    xticks = np.arange(0, len(avg_list), 1)
    # Individual lines and points
    [plt.plot(d_prime, c='black', alpha=0.25) for d_prime in mega_d] 
    [plt.scatter(x=xticks,y=d_prime, c='black', alpha=0.3) for d_prime in mega_d]  
    
    # Average
    plt.plot(avg_list, c='black', linewidth=2)
    plt.scatter(x=xticks, y=avg_list, c='black', linewidths=2)

    # SEM
    sem_list = np.array(std_list)/len(mouse_data)
    if len(mouse_data) > 1:
        plt.errorbar(xticks, avg_list, yerr=sem_list, c='black',capsize=5)

    # Format
    plt.ylim([-0.05, np.max(avg_list)+0.1])
    plt.ylabel('d\' (Sensitivity Index)')
    plt.xticks(xticks, xticks+1)
    plt.xlabel('Session')
    plt.title(mouse.id)

    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save }')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600) 
    plt.show()


def plot_falsePositives(mouse_data, save=False, max_n=10, skip_first_n=0):
    # Not very necessary plot tbh
    """
    Plot the progression of false positives (successes on catch trials) across sessions for a given mouse.

    Parameters:
    - mouse_data: A mouse object with .sessions and .all_data attributes.
    - max_n (int): Size of the running average window.
    - skip_first_n (int): Number of initial trials to ignore when plotting.
    
    Returns:
    - FP_dict: Dictionary of false positive average scores per session.
    """
    # Extract some information we are to use later
    n_sessions = len(mouse_data.sessions)
    FP_dict = {}
    max_trial_lengths = []

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")  # Dynamic colors

    for sesh_idx, session in enumerate(mouse_data.sessions):
        data = mouse_data.session_data[session]

        # Get catch trials
        catch_trials = select_trialType(data, 2)
        max_trial_lengths.append(len(catch_trials))

        # Running calculation
        FP_scores = []
        running_window = deque(maxlen=max_n)
        total_trials = hits = 0

        for _, trial in catch_trials.iterrows():
            total_trials += 1
            if trial['success']:
                hits += 1
            score = (hits / total_trials) * 100  # FP percentage
            running_window.append(score)
            FP_scores.append(np.mean(running_window))

        # Skip first N if specified
        FP_avg_trimmed = FP_scores[skip_first_n:]

        # Normalize x-axis to 0–100%
        # x_vals = np.linspace(0, 100, len(FP_avg_trimmed)) # Plot the percentage of completed catch trials as x-axis
        x_vals = np.arange(skip_first_n, skip_first_n + len(FP_avg_trimmed)) # PLot actual X trials
        ax.plot(x_vals, FP_avg_trimmed, label=session, color=cmap(sesh_idx % 10))
        FP_dict[sesh_idx] = FP_avg_trimmed

    # Plot formatting
    ax.set_title(f'Progression of False Positives \n{mouse_data.id}')
    # ax.set_xlabel('Completed Catch Trials (%)')
    ax.set_xlabel('Catch trial #')
    ax.set_ylabel('False Positive Rate (%)')
    ax.legend()
    plt.tight_layout()
    
    # Save
    if save:
        save = Path(save)
        os.makedirs(save.parent, exist_ok=True)
        print(f'Saving to: {save }')
        fig.savefig(save.with_suffix('.svg'), bbox_inches='tight', dpi=600)
        fig.savefig(save.with_suffix('.jpg'), bbox_inches='tight', dpi=600) 
    plt.show()

    return 
    
# Group plots showing differences between conditions
def plot_d_prime_comparison(*args, legend_names=None, colors=None):
    ''' Plots the d prime of each session for experimental groups

    INPUT:
        *args: the mouse_data lists of each experimental condition i.e. ctrl_data, exp_data etc
        legend_names (list): a list of strings that descibe the experimental groups i.e. saline, rapamycin
        colors (list): list of strings that define the color of each condition
    OUTPUT:
        a beautiful plot
    '''
    if not legend_names:
        legend_names = [f'Group {i+1}' for i in range(len(args))]
    if not colors:
        colors = ['red', 'blue', 'green', 'purple', 'orange']  # Default colors
    
    all_d_prime_lists = []
    
    for data in args:
        d_prime_lists = []
        for mouse in data:
            d_prime_list = calc_d_prime(mouse)
            d_prime_lists.append(d_prime_list)
        all_d_prime_lists.append(d_prime_lists)

    # Calculate average and standard error of the mean across sessions
    avg_lists = [np.mean(d_prime_lists, axis=0) for d_prime_lists in all_d_prime_lists]
    std_lists = [np.std(d_prime_lists, axis=0) for d_prime_lists in all_d_prime_lists]
    sem_lists = [std_list / len(data) for std_list, data in zip(std_lists, args)]

    # Plotting
    fig = plt.figure(figsize=(6, 4))
    
    for i, (avg_list, color, legend_name) in enumerate(zip(avg_lists, colors, legend_names)):
        # Individual lines and points for each mouse in the group
        for d_prime_list in all_d_prime_lists[i]:
            plt.plot(d_prime_list, c=color, alpha=0.25)
            plt.scatter(np.arange(len(d_prime_list)), d_prime_list, c=color, alpha=0.3)
        
        # Average line and point for the group
        plt.plot(avg_list, c=color, linewidth=3, label=legend_name)
        plt.scatter(np.arange(len(avg_list)), avg_list, c=color, linewidths=3)
        
        # Standard error of the mean (SEM) error bars
        plt.errorbar(np.arange(len(avg_list)), avg_list, yerr=sem_lists[i], c=color, capsize=5)
    
    # Format plot
    plt.ylim([-0.05, 5])
    plt.ylabel('d\' (Sensitivity Index)')
    plt.xlabel('Session')
    plt.xticks(np.arange(len(avg_list)), np.arange(1, len(avg_list) + 1))
    plt.legend(fontsize=10)
    plt.title('Sensitivity Index (d\') by Session')
    plt.show()


def plot_session_rt_comparison(*args, legend_names=None, colors=None):
    """Plot response time averages and error bars over sessions."""
    if not legend_names:
        legend_names = [f'Group {i+1}' for i in range(len(args))]
    if not colors:
        colors = ['red', 'blue', 'green', 'purple', 'orange']  # Default colors
    
    # Calculate the rt of each group
    all_rt_lists = []

    for data in args:
        # Calculate the group's response time
        rt_list = get_RTs(data, trial_type=1)
        all_rt_lists.append(rt_list)
    
    # Calculate average and standard error of the mean across sessions
    avg_lists = [np.mean(rt_list, axis=0) for rt_list in all_rt_lists]
    std_lists = [np.std(rt_list, axis=0) for rt_list in all_rt_lists]
    sem_lists = [std_list / len(data) for std_list, data in zip(std_lists, args)]

    # Plotting
    fig = plt.figure(figsize=(6, 4))
    
    for i, (avg_list, color, legend_name) in enumerate(zip(avg_lists, colors, legend_names)):
        # Individual lines and points for each mouse in the group
        for rt_list in all_rt_lists[i]:
            plt.plot(rt_list, c=color, alpha=0.25)
            plt.scatter(np.arange(len(rt_list)), rt_list, c=color, alpha=0.3)
        
        # Average line and point for the group
        plt.plot(avg_list, c=color, linewidth=3, label=legend_name)
        plt.scatter(np.arange(len(avg_list)), avg_list, c=color, linewidths=3)
        
        # Standard error of the mean (SEM) error bars
        plt.errorbar(np.arange(len(avg_list)), avg_list, yerr=sem_lists[i], c=color, capsize=5)
    
    # Format plot
    plt.ylim([-0.05, 1.5])
    plt.ylabel('Response Time (s)')
    plt.xlabel('Session')
    plt.xticks(np.arange(len(avg_list)), np.arange(1, len(avg_list) + 1))
    plt.legend(fontsize=10)
    plt.show()


def plot_PLick_comparison(*args, legend_names=None, colors=None):
    ''' docstring
    '''
    if not legend_names:
        legend_names = [f'Group {i+1}' for i in range(len(args))]
    if not colors:
        colors = ['red', 'blue', 'green', 'purple', 'orange']  # Default colors
    
    # Calculate the Plick of each group
    all_PLick_lists = []
    for data in args:
        PLick_list = []
        for mouse in data:
            # Calculate the group's response time
            PLick, _ = get_PLick(mouse)
            PLick_list.append(PLick)
        all_PLick_lists.append(PLick_list)
    
    # Calculate average and standard error of the mean across sessions
    avg_lists = [np.mean(PLick, axis=0) for PLick in all_PLick_lists]
    std_lists = [np.std(PLick, axis=0) for PLick in all_PLick_lists]
    sem_lists = [std_list / len(data) for std_list, data in zip(std_lists, args)]

    # Plotting
    fig = plt.figure(figsize=(6, 4))
    
    for i, (avg_list, color, legend_name) in enumerate(zip(avg_lists, colors, legend_names)):
        # Individual lines and points for each mouse in the group
        for PLick_list in all_PLick_lists[i]:
            plt.plot(PLick_list, c=color, alpha=0.25)
            plt.scatter(np.arange(len(PLick_list)), PLick_list, c=color, alpha=0.3)
        
        # Average line and point for the group
        plt.plot(avg_list, c=color, linewidth=3, label=legend_name)
        plt.scatter(np.arange(len(avg_list)), avg_list, c=color, linewidths=3)
        
        # Standard error of the mean (SEM) error bars
        plt.errorbar(np.arange(len(avg_list)), avg_list, yerr=sem_lists[i], c=color, capsize=5)
    
    # Format plot
    plt.ylim([-0.05, 1.05])
    plt.ylabel('P(lick)')
    plt.xlabel('Session')
    plt.xticks(np.arange(len(avg_list)), np.arange(1, len(avg_list) + 1))
    plt.legend(fontsize=10)
    plt.show()


# Legacy plots
def plot_cumScore(mouse_data):
    ''' Takes a list of mouse_data, calculates their individual and average cumulative score and plots it
    '''
    # TODO Should be function for individual animal, experimental group and full experiment
    # Thus mouse_data = Mouse_Data, [Mouse_Data] and [[Mouse_Data], [Mouse_Data]]

    # Calculate the cumulative score
    cum_scores = [] 
    for mouse in mouse_data:
        cum_score = get_cum_score(mouse)
        cum_scores.append(cum_score)
    cum_scores = extend_lists(cum_scores)

    data = get_average_cum_score(cum_scores)

    # Now plot
    # Set figure basics 
    fig = plt.figure(figsize=(16,6))
    fig.patch.set_facecolor('white')
    
    # First plot the average lines
    ctrl_line = plt.plot(data['avg'], linewidth=3, color='black')
    # plt.legend(['Control', 'Anisomycin'])

    # Add SEM
    # Control
    y_min = np.subtract(data['avg'], data['sem'])
    y_max = np.add(data['avg'], data['sem'])
    x = np.arange(0, len(data['avg']), 1)
    plt.fill_between(x, y_min, y_max, alpha=0.5, color='black')

    # Individual lines
    for trace in data['raw']:
        plt.plot(trace, c='gray')

    plt.show()
    return

def plot_groupCumScore(ctrl_data, exp_data):
    ''' Takes a list of mous_data, calculates their individual and average cumulative score and plots it
    '''
    # TODO Should be function for individual animal, experimental group and full experiment
    # Thus mouse_data = Mouse_Data, [Mouse_Data] and [[Mouse_Data], [Mouse_Data]]

    # Calculate the cumulative score
    ctrl_cum_scores = [] 
    for mouse in ctrl_data:
        cum_score = get_cum_score(mouse)
        ctrl_cum_scores.append(cum_score)
    ctrl_cum_scores = extend_lists(ctrl_cum_scores)
    ctrl_data = get_average_cum_score(ctrl_cum_scores)

    # Experimental group
    exp_cum_scores = [] 
    for mouse in exp_data:
        cum_score = get_cum_score(mouse)
        exp_cum_scores.append(cum_score)
    exp_cum_scores = extend_lists(exp_cum_scores)
    exp_data = get_average_cum_score(exp_cum_scores)


    # Now plot
    # Set figure basics 
    fig = plt.figure(figsize=(16,6))
    fig.patch.set_facecolor('white')

    # Add SEM
    # Control
    y_min = np.subtract(ctrl_data['avg'], ctrl_data['sem'])
    y_max = np.add(ctrl_data['avg'], ctrl_data['sem'])
    x = np.arange(0, len(ctrl_data['avg']), 1)
    plt.fill_between(x, y_min, y_max, alpha=0.5, color='black')

    # Individual lines
    for trace in ctrl_data['raw']:
        plt.plot(trace, c='gray', alpha=0.5)

    # Experimental group
    # Add SEM
    y_min = np.subtract(exp_data['avg'], exp_data['sem'])
    y_max = np.add(exp_data['avg'], exp_data['sem'])
    x = np.arange(0, len(exp_data['avg']), 1)
    plt.fill_between(x, y_min, y_max, alpha=0.5, color='red')

    for trace in exp_data['raw']:
        plt.plot(trace, c='red', alpha=0.5)

     # First plot the average lines
    ctrl_line = plt.plot(ctrl_data['avg'], linewidth=3, color='black', label='Control')
    exp_line = plt.plot(exp_data['avg'], linewidth=3, color='red', label='Rapamycin')
    # plt.legend([ctrl_line, exp_line],['Control', 'Rapamycin'])
    plt.legend()


    # Make pretty
    plt.ylabel(r'$\sum$ hits - misses')
    plt.xlim([0, 300])
    # plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.show()

    return

