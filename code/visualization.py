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
from matplotlib.gridspec import GridSpec
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
    print(mouse.sessions)
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
        fname = input('What path do you want to save the .jpg to?')
        fig.savefig(fname+mouse_data.id+'.jpg', bbox_inches='tight')   
    else:   
        plt.show()
    return


def plot_performance(mouse_data, average=False):
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
                plt.scatter(x=n_session-0.2, y=hit_rate, color='blueviolet', zorder=10, edgecolors='black', linewidths=0.5, s=10)
                plt.scatter(x=n_session, y=FP_rate, color='gray', zorder=10, edgecolors='black', linewidths=0.5, s=10)
                plt.scatter(x=n_session+0.2, y=effect_rate, color='plum', zorder=10, edgecolors='black', linewidths=0.5, s=10)
            
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
            axs.bar([n_session-0.2, n_session, n_session+0.2], [hit_rate, FP_rate, effect_rate], color=['blueviolet', 'gray', 'plum'], width=0.2, yerr=[hit_std, FP_std, effect_std], capsize=2)

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
            axs.bar([n_session-0.2, n_session, n_session+0.2], [hit_rate, FP_rate, effect_rate], color=['blueviolet', 'gray', 'plum'], width=0.2)
            
            
    # Formatting
    fig.suptitle(title, y=.925)  
    axs.set_ylabel('Percentage (%)')
    axs.set_ylim([0, 110])

    axs.set_xticks(xticks)
    axs.set_xticklabels([1,2,3,4])
    axs.set_xlabel('Session')
    
    # Add legend
    blue_patch = matplotlib.patches.Patch(color='blueviolet', label='Hit')
    red_patch = matplotlib.patches.Patch(color='gray', label='FP')
    plum_patch = matplotlib.patches.Patch(color='plum', label='Hit - FPs')
    axs.legend(bbox_to_anchor=(0.005, 0.89, 1., .102), handles=[blue_patch, red_patch, plum_patch], loc='upper left', borderaxespad=0., ncol=1)

    plt.show()
    return


def plot_d_prime(mouse_data):
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
    plt.ylim([-0.05,5])
    plt.ylabel('d\' (Sensitivity Index)')
    plt.xticks([0,1,2,3], [1,2,3,4])
    plt.xlabel('Session')
    plt.show()


def plot_lickPerformance(mouse_data, save=False, peak=False):
    '''docstring'''
    # Check for peaking allowing the user to only see the plots of the first 2 sessions
    if peak:
        n_sessions = 2
    else:
        n_sessions = len(mouse_data.sessions)

    # Set figure basics 
    fig, axs = plt.subplots(1, n_sessions, figsize=(15, 10)) # Size plot according to the number of sessions
    plt.subplots_adjust(wspace=0.5) 
    fig.patch.set_facecolor('white')
    fig.suptitle(str(mouse_data.id), y=1.0)

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
        axs[idx].set_ylabel('Stim Trial #')
        axs[idx].set_xlabel('Time (s)')
        axs[idx].set_title(str(session))
        axs[idx].set_xticks(np.arange(0, 1.8, 0.5))

    # After all trials have been plotted adjust all plots
    plt.show()
    return


def plot_Plick(mouse):
    """
    Plot lick probability (Plick) for microstimulation ('mstimP') and catch trials ('catchP').

    Parameters:
        mouse (Mouse_Data): Mouse_Data instance containing lick probability data.

    Returns:
        None
    """
    # Get microstimulation (mstimP) and catch trial (catchP) lick probabilities
    mstimP, catchP = get_PLick(mouse)

    # Create a new figure and axis for plotting
    fig, axs = plt.subplots(figsize=(6, 8))

    # Plot microstimulation lick probability ('mstimP') in blueviolet color and ('catchP') in gray color
    axs.plot(mstimP, color='blueviolet', alpha=1, label=r'$\mu$Stim')
    axs.plot(catchP, color='gray', alpha=1, label='Catch') 
    x = np.arange(0, len(mstimP), 1)
    axs.scatter(x, mstimP, c='blueviolet')  # Scatter plot for 'stim' trials
    axs.scatter(x, catchP, c='gray')  # Scatter plot for 'catch' trials

    # Set plot title and labels
    axs.set_title(mouse.id)  # Set the plot title to mouse id (if desired)
    axs.set_xticks([0,1,2,3])
    axs.set_xticklabels([1, 2, 3, 4])  # Set custom x-axis tick labels
    axs.set_xlabel('Session')  # Set x-axis label to 'Session'
    axs.set_ylabel('P(lick)')  # Set y-axis label to 'P(lick)'

    # Set y-axis limits between -0.05 and 1.05 for a proper range of probabilities
    axs.set_ylim([-0.05, 1.05])

    # Display legend for the plot
    axs.legend(loc='upper left')
    
    # Show the plot
    plt.show()
    return


def plot_RT(mouse_data):
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

    # Ensure mouse_data is treated as a list even if it's a single instance
    if not isinstance(mouse_data, list):
        mouse_data = [mouse_data]

    # Iterate over each mouse in the list
    for mouse in mouse_data:
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

    # Prepare x-axis values (sessions)
    x = np.arange(len(stim_avg))

    # Create a new figure and axis for plotting
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot average response times and add scatter points
    ax.plot(stim_avg, label='Stim', c='blueviolet')
    ax.scatter(x, stim_avg, c='blueviolet')  # Scatter plot for 'stim' trials
    ax.plot(catch_avg, label='Catch', c='gray')
    ax.scatter(x, catch_avg, c='gray')  # Scatter plot for 'catch' trials

    # If multiple mice are plotted, show error bars representing standard deviation
    ax.errorbar(x, stim_avg, yerr=stim_std, c='blueviolet', capsize=5)
    ax.errorbar(x, catch_avg, yerr=catch_std, c='gray', capsize=5)

    # Set plot limits, ticks, labels, and display the legend
    ax.set_ylim([0, 1.2])
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xlim([-0.5, len(stim_avg) - 0.5])  # Adjust x-axis limits based on number of sessions
    ax.set_xticks(x)
    ax.set_xticklabels([1,2,3,4])
    ax.set_ylabel('Response time (s)')
    ax.set_xlabel('Session')

    # Display the plot
    plt.show()
    return

def plot_session_rt(rt_data, color='black'):
    """Plot response time averages and error bars over sessions."""
    # Get the x ticker
    num_sessions = len(rt_data[0])
    x = np.arange(num_sessions)
    n = np.shape(rt_data)[0]
    print(n)

    # Plot the average RT of every animal of every session
    for mouse_rts in rt_data: # should be a list of individual animals response times
        plt.plot(mouse_rts, color=color, alpha=0.5)
    
    # Calculate mean and std
    avg = np.mean(rt_data, 0)
    std = np.std(rt_data, 0)
    plt.plot(avg, color=color)
    plt.scatter(x, avg, color=color)
    plt.errorbar(x=x, y=avg, yerr=std, c=color, capsize=2)
    plt.ylim([0, 1.5])
    plt.yticks([0,0.5,1,1.5])
    plt.ylabel('Response Time')
    plt.show()

    
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

