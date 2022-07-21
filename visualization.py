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
matplotlib.rcParams.update({'font.size':16, 'font.family':'Times New Roman', 'axes.facecolor':'white'})   

def plot_raster_rt(mouse, save=False):
    ''' Creates a figure containing rasterplots of the trial response time.

        INPUT:
            mouse(Mouse_Data): Dataclass with attributes like id, sessions, all_data and concatenated data
            save(bool): prompts the user for destination folder path
        OUTPUT:
            raster_rt_plot(matplotlib.plt): either a plot is shown or saved
    '''
    # Set figure basics 
    fig, axs = plt.subplots(1, len(mouse.sessions), figsize=(15, 10)) # Size plot according to the number of sessions
    plt.subplots_adjust(wspace=1,) 
    fig.patch.set_facecolor('white')
    fig.suptitle(str(mouse.id), y=1.05)
    
    # Get and plot data for every session
    for idx, session in enumerate(mouse.sessions):
        colors = []
        rt_full = mouse.all_data[session]['response_t']
        x = np.arange(0, len(rt_full), 1) # Initiate x-axis for plotting
        rt = [[rt] for rt in rt_full] # plt.eventplot requires values as list to ascribe different offsets
        intensity = mouse.full_data['intensity']
        intensity = np.sort(list(set(intensity)))
        cmap = matplotlib.cm.get_cmap('summer', len(intensity)).reversed()
        
        # Pick right intensity color
        for _, trial in mouse.all_data[session].iterrows():
            trial_succes = trial['succes']
            trial_intensity = trial['intensity']
            if trial_succes:
                c_idx = np.where(intensity==trial_intensity)[0][0]
                c = cmap(c_idx)
                colors.append(c)
            else:
                colors.append('red')

        # Create legend patches
        gray_patch = matplotlib.patches.Patch(color='gray', label='Stimulus')
        red_patch = matplotlib.patches.Patch(color='red', label='Incorrect trials')
        
        # Now the plot
        if len(mouse.sessions) > 1:
            for x in np.arange(0, 0.15, 0.001):
                axs[idx].axvline(x, color='gray')
            offset = np.arange(0, len(rt), 1)
            axs[idx].eventplot(rt, lineoffsets=offset, linewidth=7.5, colors=colors)
            axs[idx].set_xlim([-0.2, 0.8])
            axs[idx].set_ylabel('Trial #')
            axs[idx].invert_yaxis()
            axs[idx].set_xlabel('Response time (s)')
            axs[idx].set_title(str(session))
            axs[idx].set_xticks(np.arange(0, 1, 0.2))
            legend = axs[0].legend(bbox_to_anchor=(0., 1.1, 1., .102), handles=[red_patch, gray_patch], mode="expand", borderaxespad=0., ncol=1)
    
        else:
            for x in np.arange(0, 0.15, 0.001):
                axs.axvline(x, color='gray')
            offset = np.arange(0, len(rt), 1)
            axs.eventplot(rt, lineoffsets=offset, linewidth=7.5, colors=colors)
            axs.set_xlim([-0.2, 0.8])
            axs.set_ylabel('Trial #')
            axs.invert_yaxis()
            axs.set_xlabel('Response time (s)')
            axs.set_title(str(session))
            legend = axs.legend(bbox_to_anchor=(0., 1.1, 1., .102), handles=[red_patch, gray_patch], mode="expand", borderaxespad=0., ncol=1)
    
    # Add colormap
    cmap_bar = matplotlib.cm.get_cmap('summer', len(intensity))
    cmappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1), cmap=cmap_bar.reversed())
    cax = [0.3, 0.96, 0.5, .05]
    cax = fig.add_axes(cax, snap=False)
    cbar = plt.colorbar(cmappable, cax=cax, ticks=[0, 1], orientation='horizontal')
    cbar.ax.set_xticklabels(['Correct trial low intensity', 'Correct trial high intensity'])
    
    # Prompt user for destination folder path or show the plot
    if save:
        fname = input('What path do you want to save the .jpg to?')
        fig.savefig(fname+mouse.id+'.jpg', bbox_inches='tight')   
    else:   
        plt.show()
    return