''' video_helpers.py
This file contains the code to prepare videos for DeepLabCut (DLC) analysis. 
Please go through the corresponding jupyter notebook file before utilizing it
yourself. This is to assure you understand what is going on and make sure you 
run in to as little problems as possible

@Mik Schutte & Jelte de Vries [AG Larkum]
'''

import numpy as np
import os, cv2, re

def grabReferenceFrame(videoPath, destfolder):
    ''' Opens a videofile, grabs the first frame and saves it as a reference for 
        snipping. To be used in conjunction with ImageJ for determining x & 
        y-coordinates used in functions like split_and_flip() and crop_for_pupil().
        NOTE that if the camera is moved extensively during the video this 
        reference frame is no longer valid.
       
       INPUT: 
            videoPath(str): path to the video file you want to create a image
                            reference for.
            destfolder(str): path to the destination folder where you want to save
                             the image to.
       
       OUTPUT:
            referenceFrame: A .jpg file located at the destfolder.
    '''
    # Get filename and create the destfolder if needed
    filename = videoPath.split('/ | \\')[-1].split('.')[0]
    if destfolder[-1] is not '/ | \\':
        destfolder = destfolder+'/'
    os.makedirs(destfolder, exist_ok=True)

    # Open video and read frames
    capr = cv2.VideoCapture(videoPath)
    if capr.isOpened():
        succes, frame = capr.read()
        if succes:
            cv2.imwrite(destfolder+filename+'_ref.jpg', frame)
        else:
            print(f'FAILED to create a reference frame for {filename}')
    capr.release()
    cv2.destroyAllWindows()
    return


def splitVideo(videoPath, x_snip, destfolder, fps=200, flip=False):
    #TODO Finalize the docstring
    ''' Gets a videofile from path, reads, splits and flips each frame.

        INPUT: 
            path_og_video(str) = path to the video file you want to process
            x_snip(int) = reference frame's x-coordinate of where you want to snip
            y_snip(int) = reference frame's y_coordinate of where you want to snip  

        OUTPUT:
            2 prepped .mp4 file located at the path_og_video/prepped folder         
    '''
    # Get filename and create a destination folder
    filename = videoPath.split('/ | \\')[-1].split('.')[0]
    filetype = videoPath.split('/ | \\')[-1].split('.')[-1]
    os.makedirs(destfolder, exist_ok=True)

    # Read frames from the original video and get framesize
    capr = cv2.VideoCapture(videoPath)
    succes, frame = capr.read()
    if not succes:
        print(f'FAILED unable to open video')
    og_height, og_width, _ = frame.shape
    capr.release()


    # Get dimensions for fourcc
    cropped_img = frame[0:og_height, 0:x_snip]
    cropped_img2 = frame[0:og_height, x_snip:og_width]
    height, width, _ = cropped_img.shape
    height2, width2, _ = cropped_img2.shape

    # Instantiate four-character code & Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(destfolder+filename+'_left.'+filetype, fourcc, fps, (width, height))
    output2 = cv2.VideoWriter(destfolder+filename+'_right.'+filetype, fourcc2, fps, (width2, height2))

    # Read and split
    capr = cv2.VideoCapture(videoPath)
    framecount = int(capr.get(cv2.CAP_PROP_FRAME_COUNT))
    # Write to newfile
    print(f'Writing snipped video: {filename}')
    iFrame = 0
    while True:
        iFrame += 1
        success, frame = capr.read()
        if success:
            cropped_img = frame[0:og_height, 0:x_snip]
            if flip:
                cropped_img2 = cv2.flip(frame[0:og_height, x_snip:og_width], 1)
            else:
                cropped_img2 = frame[0:og_height, x_snip:og_width]
            output.write(cropped_img)
            output2.write(cropped_img2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break 

    # Finish up
    print(f'Finished snipping {filename}')
    if framecount != iFrame:
        print(f'WARNING the number of frames is not equal to the original video!')
    capr.release()
    cv2.destroyAllWindows()
    return
