# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:11:13 2023

@author: Anne
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:25:45 2023

@author: Anne
"""


#!/usr/bin/env python

"""
Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.
The demo demonstrates how to use the params, MotionCorrect and cnmf objects
for processing 1p microendoscopic data. The analysis pipeline is similar as in
the case of 2p data processing with core difference being the usage of the
CNMF-E algorithm for source extraction (as opposed to plain CNMF). Check
the companion paper for more details.

You can also run a large part of the pipeline with a single method
(cnmf.fit_file) See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline_cnmfE.ipynb)
"""

import logging
import matplotlib.pyplot as plt
import numpy as np

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params

import os #For all the file path manipulations
from time import time # For time logging
from glob import glob
#import sys #To apppend the python path for the selection GUI
#sys.path.append('C:/Users/Anne/Documents/chiCa') #Include the path to the functions
#import neuron_selection_GUI

# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)
    # filename="/tmp/caiman.log"

##%%
def main():
    pass # For compatibility between running under Spyder and the CLI

    #  start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=8,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)
    
    #  First setup some parameters for motion correction
        # dataset dependent parameters
    #    fnames = ['data_endoscope.tif']  # filename to be processed
    #    fnames = [download_demo(fnames[0])]  # download file if not already present
    #    filename_reorder = fnames
        
        #To load all the movies in the folder
    #   from pathlib import Path
    #   import os
    #   from natsort import os_sorted
    #
    #   base_dir = Path("D:/data/LO012/20210825_164317/miniscope/binned")
    #   fnames = list(base_dir.glob('**/*.avi'))
    #   fnames = os_sorted(fnames)
    #   print(fnames)
        
    #To let user select files
    from tkinter import Tk
    from tkinter.filedialog import askopenfilenames
    Tk().withdraw() #Don't let the verification window pop up
    fileSet = askopenfilenames(title = "Select your imaging movie files",
                            filetypes =[("AVI files","*.avi*")])
    fnames = list(fileSet)
        
    #
    
    fr = 30                          # movie frame rate
    decay_time = 0.6                 # length of a typical transient in seconds
    
    # motion correction parameters
    motion_correct = True            # flag for motion correction
    pw_rigid = False                 # flag for pw-rigid motion correction
    
    gSig_filt = (3, 3)   # size of filter, in general gSig (see below),
    #                      change this one if algorithm does not work
    max_shifts = (5, 5)  # maximum allowed rigid shift
    strides = (48, 48)   # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    border_nan = 'copy'
    
    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }
    
    opts = params.CNMFParams(params_dict=mc_dict)
    
    #  MOTION CORRECTION
    #  The pw_rigid flag set above, determines where to use rigid or pw-rigid
    #  motion correction
    if motion_correct:
        #Track the time to compute shifts
        mc_start_time = time()
        
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
            plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
            plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')
    
        bord_px = 0 if border_nan == 'copy' else bord_px
        #fname_mc = [os.path.join(os.path.split(individual_movie)[0], 'motion_corrected_movie.mmap') for individual_movie in mc.fname]
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
        
        mc_end_time = time()
        #Display elapsed time
        print(f"Motion correcition finished in {round(mc_end_time - mc_start_time)} s.")
        
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(filename_reorder, base_name='memmap_',
                                   order='C', border_to_0=0, dview=dview)
    #-- Save the shifts, the very basic implementation    
    rigid_shifts = np.array(mc.shifts_rig) # Retrieve shifts from mc object
    
    outputPath = os.path.dirname(fnames[0]) #Assuming that you want the results in the same location
    np.save(outputPath + '/rigid_shifts', rigid_shifts) #Save the np array to npy file

    #Do a round of clean-up before the uploading the results
    for f in fname_mc:
        os.remove(f) #Delete all the originally generated motion corrected movie files in fortran order

    c_files = glob(os.path.join(outputPath, '*.mmap'))
    c_files.remove(fname_new) #Remove the concatenated motion corrected file from the list of sub-movies
    for f in c_files:
        os.remove(f) #Now also delete all the C ordered files that annoyingly also get created with the last call to save_mmap!

    #
    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new, mode='r+')
    images = Yr.T.reshape((T,) + dims, order='F')
        
    
    # Parameters for source extraction and deconvolution (CNMF-E algorithm)
    
    p = 1               # order of the autoregressive system
    K = None            # upper bound on number of components per patch, in general None for 1p data
    gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
    Ain = None          # possibility to seed with predetermined binary masks
    merge_thr = .7      # merging threshold, max correlation allowed
    rf = 80             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20    # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2            # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1            # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0             # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0        # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = .7       # min peak value from correlation image
    min_pnr = 8        # min peak to noise ration from PNR image
    ssub_B = 4          # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor
    
    opts.change_params(params_dict={'dims': dims,
                                    'method_init': 'corr_pnr',  # use this for 1 photon
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,    # set it to True to run CNMF-E
                                    'nb': gnb,
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                    'low_rank_background': low_rank_background,
                                    'update_background_components': True,  # sometimes setting to False improve the results
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'normalize_init': False,               # just leave as is
                                    'center_psf': True,                    # leave as is for 1 photon
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'del_duplicates': True,                # whether to remove duplicates from initialization
                                    'border_pix': bord_px})                # number of pixels to not consider in the borders)
    
    #Remove pixels with basically zero intensity but very few 
    
    medProj = np.median(images, axis=0, keepdims=True)
    median_bool = np.squeeze(medProj < 1)
    for k in range(images.shape[0]):
        temp = images[k,:,:]
        temp[median_bool] = 0.0001
        images[k,:,:] = temp
            
    #  compute some summary images (correlation and peak to noise)
    # change swap dim if output looks weird, it is a problem with tiffile
    corr_image_start = time()
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::10], gSig=gSig[0], swap_dim=False)
    #compute the correlation and pnr image on every frame. This takes longer but will yield
    #the actual correlation image that can be used later to align other sessions to this session
    corr_image_end = time()
    print(f"Computed correlation- and pnr images in {corr_image_end - corr_image_start} s.")
    
    np.save(outputPath + '/spatio_temporal_correlation_image', cn_filter)
    np.save(outputPath + '/median_projection', medProj)
    # if your images file is too long this computation will take unnecessarily
    # long time and consume a lot of memory. Consider changing images[::1] to
    # images[::5] or something similar to compute on a subset of the data
    
    # inspect the summary images and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)
    
    # Shut donw shut down parallel pool and restart if desired
    dview.terminate()
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=4,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)
    # RUN CNMF ON PATCHES
    cnmfe_start_time = time()
    
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)
    cnm.estimates.detrend_df_f() #Also reconstruct the detrended non/denoised trace
    dview.terminate()
    
    #Display elapsed time
    cnmfe_end_time = time()
    print(f"Ran initialization and fit cnmfe model in {round(cnmfe_end_time - cnmfe_start_time)} s.")
    
    # Save first round of results
    cnm.save(outputPath + '/uncurated_caiman_results.hdf5')
    
    #-----------------------------------------------------------------------------
    ###############################################################################
    #%% Set up and runfunctions for the GUI for manual curation
    
   #Function to get the caiman estimates, especially a full size and original dimension A
    def retrieve_caiman_estimates(data_source):
       '''Fetch cnmfe outputs either from a saved h5 file or from a caiman
       object directly.
       
       Parameters
       ----------
       data_source: Name of h5 file or caiman object
       
       Returns
       ------
       
       Usage
       ----------
       A, C, S, F, image_dims, frame_rate, neuron_num, recording_length, movie_file, spMat = load_caiman_estimates(data_source)
       ----------------------------------------------------------------------------
       '''
       
       import numpy as np
       import h5py
       import scipy
       import scipy.sparse
       from pathlib import Path
    
       # Determine the data source - either hdf5 file or caiman object   
       if isinstance(data_source, str): #Loading the data from HDF5    
            hf = h5py.File(data_source, 'r') #'r' for reading ability
    
        # Extract the noisy, extracted and deconvolved calcium signals
        # Use the same variable naming scheme as inside caiman
    
            params = hf.get('params') 
            image_dims = np.array(params['data/dims'])
            frame_rate = np.array(params['data/fr'])
            movie_file = hf.get('mmap_file')[()] # Use [()] notation to access the value of a dataset in h5
            if not isinstance(movie_file, str):
                movie_file = movie_file.decode() #This is an issue when changing to other operating system
            movie_file = Path(movie_file) #Convert to path
        
            C = np.array(hf.get('estimates/C'))
            S = np.array(hf.get('estimates/S'))
            
            try:
                F = np.array(hf.get('estimates/F_dff'))
            except:
                F = None
            
            # Get the sparse matrix with the shapes of the individual neurons
            temp = hf.get('estimates/A')
            
            # Reconstruct the sparse matrix from the provided indices and data
            spMat = scipy.sparse.csc_matrix((np.array(temp['data']),np.array(temp['indices']),
                                np.array(temp['indptr'])), shape=np.array(temp['shape']))
            
       else: #Directly accessing from caiman object
            image_dims = data_source.dims
            frame_rate = data_source.params.data['fr']
            movie_file = data_source.mmap_file
            movie_file = Path(movie_file) #Convert to path
            
            
            C = data_source.estimates.C
            S = data_source.estimates.S
            if data_source.estimates.F_dff is None:
                F = None
            else:
                F = data_source.estimates.F_dff
            
            spMat = data_source.estimates.A
            
        # Retrieve other useful info from the shape of the signal 
       neuron_num = C.shape[0]
       recording_length = C.shape[1]
    
       deMat = np.array(spMat.todense()) # fill the holes and transform to numpy array
    
        # Several important things here: Other than in caiman the output is saved as 
        # n x neuron_num matrix. Therefore, the neuron dimension will be the third one.
        # Also, it is important to set the order of reshaping to 'F' (Fortran).
       A = deMat.reshape(image_dims[0], image_dims[1], neuron_num, order='F')
    
        #---Define the outputs
       print('-------------------------------')
       print(f'Loaded data with {F.shape[0]} detected putative neurons')
       return A, C, S, F, image_dims, frame_rate, neuron_num, recording_length, movie_file, spMat
       #---------------------------------------------------------------
       
    #Define the GUI for the interactive neuron selection
    def neuron_selection_GUI(data_source = None):
        '''run_current_neuron(data_source = None)
        run_current_neuron(data_source = 'C:Users/Documents/analyzedData.hdf5')
        run_current_neuron(data_source = caiman_object)
        This function allows the user to go through individually identified
        putative neurons identified with CNMF-E and select good and discard bad
        ones. Accepted inputs are path string to saved outputs as HDF5 or caiman
        objects directly. When called without arguments the user is able to
        select a saved file. The neurons are presented according to their maximum
        fluorescence intensity value, such that bright components are shown first
        and dimmer "noise" or background components last. Unclassified components
        will be treated as discarded.'''
        #--------------------------------------------------------------------
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider #For interactive control of frame visualization
        from matplotlib.gridspec import GridSpec #To create a custom grid on the figure     
        
        #-------Declare global variables and specify the display parameters 
        global neuron_contour
        global current_neuron #The neuron currently on display
        global accepted_components #List of booleans determining whether to keep a cell or not
        global keep_neuron #List of the indices of the cell that will be kept in the end
            
        current_neuron = 0 #Start with the first neuron
        display_window = 30 # in seconds 
    
        # Load  retrieve the data and load the binary movie file
        A, C, S, F, image_dims, frame_rate, neuron_num, recording_length, movie_file, spatial_spMat = retrieve_caiman_estimates(data_source)
        
        # Load the motion corrected movie (memory mapped)
        Yr = np.memmap(movie_file, mode='r', shape=(image_dims[0] * image_dims[1], recording_length),
                    order='C', dtype=np.float32)
        # IMPORTANT: Pick the C-ordered version of the file and specify the dtype as np.float32 (!!!)
        movie = Yr.T.reshape(recording_length, image_dims[0], image_dims[1], order='F') # Reorder the same way as they do in caiman
        del Yr # No need to keep this one...
            
        #----Initialize the list of booleans for good neurons and a list with the indices    
        accepted_components = [None] * neuron_num # Create a list of neurons you want to keep or reject
        keep_neuron = [None] #Initialize the output of indices of the neurons to refit
        
        #----Sort the cells according to maximum instensity
        intensity_maxima = np.max(F,1) #Get maximum along the second dimension, e.g. within each row
        idx_max_int = np.argsort(-intensity_maxima) #The negative sign make the sorting go in descending order
        
        #Sort the data accordingly
        C = C[idx_max_int,:]
        S = S[idx_max_int,:]
        F = F[idx_max_int,:]
        A = A[:,:,idx_max_int]
    
        #--Function to prepare display range for plotted traces and the frame number according to the given time
        def adjust_display_range(display_time, display_window, frame_rate, recording_length):
            display_frame = int(display_time * frame_rate)
            frame_window = display_window * frame_rate
            frame_range = np.array([np.int(display_frame-frame_window/2), np.int(display_frame+frame_window/2)])
            #Handle the exceptions where the display cannot be symmetric, start and end of the trace
            if frame_range[0] < 0:
                frame_range[0] = 0
                frame_range[1] = frame_window
            elif frame_range[1] > recording_length:
                frame_range[0] = recording_length - display_window * frame_rate
                frame_range[1] = recording_length
                    
            return display_frame, frame_range
                
        #-------Prepare the plot
        fi = plt.figure(figsize=(14,9))
        gs = GridSpec(nrows=4, ncols=4, height_ratios=[6,3.5,0.1,0.4], width_ratios=[5,3,0.5,1.5])
        fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
        
        movAx = fi.add_subplot(gs[0,0:1]) 
        movAx.xaxis.set_visible(False) #Remove the axes where not needed
        movAx.yaxis.set_visible(False)
        movAx.set_title("Raw movie", fontsize = 14)
        
        maskAx = fi.add_subplot(gs[0,1:4])
        maskAx.xaxis.set_visible(False) #Remove the axes where not needed
        maskAx.yaxis.set_visible(False)
        maskAx.set_title("Individual neuron denoised", fontsize = 14)
        
        traceAx = fi.add_subplot(gs[1,0:2])
        traceAx.set_xlabel('time (s)', fontsize=12)
        traceAx.set_ylabel('Fluorescence intensity (A.U.)', fontsize=12) 
        traceAx.tick_params(axis='x', labelsize=12)
        traceAx.tick_params(axis='y', labelsize=12)
        
        sliBarAx = fi.add_subplot(gs[3,0:2])
        sliBarAx.xaxis.set_visible(False) #Remove the axes where not needed
        sliBarAx.yaxis.set_visible(False)
        
        interactionAx = fi.add_subplot(gs[1,2:4])
        interactionAx.xaxis.set_visible(False) #Remove the axes where not needed
        interactionAx.yaxis.set_visible(False)
        
        #----Start plotting
        #First find the time of peak activity
        display_frame = int(np.where(C[0] == C[0].max())[0])
        # Very annoying transformation of formats:
            #First find the occurence of the maximum as a tuple and access the first
            #element of this tuple, which is an array and needs to be turned into an int!
        display_time = display_frame/frame_rate
        
        #Call the function to prepare display range
        display_frame, frame_range = adjust_display_range(display_time, display_window, frame_rate, recording_length)
        
        #First the corrected movie with contour
        movie_frame = movAx.imshow(movie[display_frame,:,:], cmap='gray', vmin=0, vmax=np.max(movie)) #FIxate the display range here
        neuron_mask = A[:,:,current_neuron] > 0 #Theshold to get binary mask
        neuron_contour = movAx.contour(neuron_mask, linewidths=0.5) #Overlay binary mask on movie frame
        
        #Then plot the de-noised cell activity alone
        pixel_intensity_scaling = 1/np.max(A[:,:,current_neuron])
        max_acti = np.max(C[current_neuron]) #Find the maximum of the denoised
        mask_image = maskAx.imshow(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]),
                                cmap='gray', vmin=0, vmax=max_acti)
        #Also take the positive value for A to make sure it is bigge
    
        #Set up the plots for the traces
        time_vect = np.linspace(frame_range[0]/frame_rate, frame_range[1]/frame_rate, np.int(frame_range[1]-frame_range[0])) 
        F_line = traceAx.plot(time_vect, F[current_neuron, frame_range[0]:frame_range[1]], label='Raw fluorescnece trace')
        S_line = traceAx.plot(time_vect, S[current_neuron, frame_range[0]:frame_range[1]], label='Estimated calcium transients')
        vert_line = traceAx.axvline(display_time, color='red')
        traceAx.grid()
        plt.setp(traceAx, xlim=(frame_range[0]/frame_rate, frame_range[1]/frame_rate))
        plt.setp(traceAx, ylim=(-10, round(np.max(F[current_neuron])+5))) #Scale y axis 
        # traceAx.tick_params(axis='x', labelsize=12)
        # traceAx.tick_params(axis='y', labelsize=12)
        # traceAx.xaxis.label.set_size(12)
        # traceAx.yaxis.label.set_size(12)
        traceAx.legend(prop={'size': 12})      
        
        # Now the text display
        # Static
        interactionAx.text(0.05,0.8,'Accept neuron:', fontsize = 12)
        interactionAx.text(0.05,0.7,'Discard neuron:', fontsize = 12)
        interactionAx.text(0.05,0.6,'Forward:', fontsize = 12)
        interactionAx.text(0.05,0.5,'Backward:', fontsize = 12)
        
        interactionAx.text(0.75,0.8,'c', fontsize = 12, fontweight = 'bold')
        interactionAx.text(0.75,0.7,'x', fontsize = 12, fontweight = 'bold')
        interactionAx.text(0.75,0.6,'>', fontsize = 12, fontweight = 'bold')
        interactionAx.text(0.75,0.5,'<', fontsize = 12, fontweight = 'bold')
        
        show_accepted = interactionAx.text(0.5, 0.2, 'Not decided', fontweight = 'bold', fontsize = 12,
            horizontalalignment = 'center', verticalalignment = 'center',
            bbox ={'facecolor':(1,1,1),'alpha':0.9, 'pad':20})
        
        #--------Set up the slider 
        frame_slider = Slider(
            ax=sliBarAx,
            label='Time',
            valmin=0,
            valmax=recording_length/frame_rate, 
            valinit=display_time, 
            valstep=1/frame_rate) #Fix the steps to integers
    
        frame_slider.label.set_size(12)
        frame_slider.vline.set_visible(False)
        
        #--The slider callback
        # The function to be called anytime a slider's value changes
        def frame_slider_update(val):
            
            display_frame, frame_range = adjust_display_range(val, display_window, frame_rate, recording_length)
            movie_frame.set_data(movie[display_frame,:,:])
            
            mask_image.set_data(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]))
            
            time_vect = np.linspace(frame_range[0]/frame_rate, frame_range[1]/frame_rate, np.int(frame_range[1]-frame_range[0])) 
            F_line[0].set_xdata(time_vect)
            F_line[0].set_ydata(F[current_neuron, frame_range[0]:frame_range[1]])
            #Stupidly the output of a call to plot is not directly a line object but a list of line
            #objects!
        
            S_line[0].set_xdata(time_vect)
            S_line[0].set_ydata(S[current_neuron, frame_range[0]:frame_range[1]])
            
            #Make the x-axis fit 
            plt.setp(traceAx, xlim=(frame_range[0]/frame_rate, frame_range[1]/frame_rate))
        
            vert_line.set_xdata(np.array([val, val]))
            fi.canvas.draw_idle()
        
        #--Set up the key callback to switch between cells
        #Set the cell number as the parameter to be updated
        def cell_selection_update(event):
            global neuron_contour
            global current_neuron
            global keep_neuron
    
            # It's necessary here to set these to globals so that they can be redefined within the function
            
            if event.key == 'right' or event.key == 'c' or event.key == 'x':
                if event.key == 'c': #The case where we accept the putative neuron
                    accepted_components[current_neuron] = True #This marks an accepted neuron and puts an entry
                elif event.key == 'x':  #The case where we reject the cell
                    accepted_components[current_neuron] = False       
                    
                if current_neuron < neuron_num:
                        current_neuron = current_neuron+1
                        
            elif event.key == 'left':
                if current_neuron > 0: 
                        current_neuron = current_neuron-1
                                
                                
            fi.suptitle(f"Neuron number {current_neuron}", fontsize=18)
            
            #Find the maximum activation of this neuron on the trace and jump to this
            # position
            display_frame = int(np.where(F[current_neuron] == F[current_neuron].max())[0])
            display_time = display_frame/frame_rate
            display_frame, frame_range = adjust_display_range(display_time, display_window, frame_rate, recording_length)
            
            #Adjust frame slider
            frame_slider.set_val(display_time)
            
            #Jump to the respective movie frame
            #movie_frame.set_data(movie[display_frame,:,:])
            
            #Update the contour on the already displayed frame
            #Need to remove the contours first, unfortunately
            for tp in neuron_contour.collections: 
                tp.remove()
                    
            neuron_mask = A[:,:,current_neuron] > 0
            neuron_contour = movAx.contour(neuron_mask, linewidths=0.5)
    
            #Update the denoised plot
            pixel_intensity_scaling = 1/np.max(A[:,:,current_neuron])
            max_acti = np.max(C[current_neuron]) #Find the maximum of the denoised trace
            mask_image.set_data(A[:,:,current_neuron] * pixel_intensity_scaling * np.abs(C[current_neuron,display_frame]))
            mask_image.set_clim(vmin=0, vmax=max_acti)
                
            #Set the plot with the traces accordingly
            F_line[0].set_ydata(F[current_neuron, frame_range[0]:frame_range[1]])
            S_line[0].set_ydata(S[current_neuron, frame_range[0]:frame_range[1]])
            plt.setp(traceAx, ylim=(-10, round(np.max(F[current_neuron])+5))) #Scale y axis
            
            #Finally also update the slider value
            
            
            #Display whether neuron is accepted or not
            if accepted_components[current_neuron] is None: #Not yet determined
                show_accepted.set_text('Not decided')
                show_accepted.set_color((0,0,0))
                show_accepted.set_bbox({'facecolor':(1,1,1),'alpha':0.9, 'pad':20})
            elif accepted_components[current_neuron] == True: #When accepted
                show_accepted.set_text('Accepted')
                show_accepted.set_color((1,1,1))
                show_accepted.set_bbox({'facecolor':(0.23, 0, 0.3),'alpha':0.9, 'pad':20})
            elif accepted_components[current_neuron] == False:
                show_accepted.set_text('Discarded')
                show_accepted.set_color((1,1,1))
                show_accepted.set_bbox({'facecolor':(0.15, 0.15, 0.15),'alpha':0.9, 'pad':20})
                
            fi.canvas.draw_idle()
            
        #----Action when fiugure is closed 
        def on_figure_closing(event):
            global keep_neuron
            
            index_selection = [i for i, val in enumerate(accepted_components) if val] # Only keep indices of accepted neurons and undecided
            
            original_indices = idx_max_int[index_selection] #Map the sorted data back to the original indices
            original_indices = np.sort(original_indices)
            keep_neuron[0:len(original_indices)] = list(original_indices) #Transform back to list
            
            print(f"Selection completed with {len(keep_neuron)} accepted neurons")
            print('-------------------------------------------------------------')
            
        #----Implement the callbacks
        # register the update function with each slider
        frame_slider.on_changed(frame_slider_update)
    
        # register the key presses
        fi.canvas.mpl_connect('key_press_event', cell_selection_update)
        
        # Detect the closing of the figure
        fi.canvas.mpl_connect('close_event', on_figure_closing)
        
        #Block execution until the figure us closed
        plt.show(block = True)
        
        return keep_neuron, frame_slider
            #-------------------------------------------------------------------------    
        
    #-----------------------------------------------------------------------------------------------
    
    
    keep_neuron, frame_slider = neuron_selection_GUI(data_source = cnm)
     # Do the selection here. Don't advance until the selection is completed since further executions are not blocked!
    print('Please close all remaining figures to finish the analzsis.')  
    #%% Now convert the list of accepted components to indices
        #keep_idx = [i for i, val in enumerate(neurons_to_keep) if val] #Returns the index of every true entry in the list
        
        #Plug the indices in before re-running the fitting prodecure
    cnm.estimates.select_components(idx_components = keep_neuron)
     
    #We don't usually refit the model after curation...
    #c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
    #                                                 n_processes=6,  # number of process to use, if you go out of memory try to reduce this one
    #                                                 single_thread=False)   
    # cnm.refit(images, dview = dview)
    #cnm.estimates.detrend_df_f() #Also reconstruct the detrended non/denoised trace
    cnm.save(outputPath + '/caiman_results.hdf5') 
    cm.stop_server(dview=dview)

# %% This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
