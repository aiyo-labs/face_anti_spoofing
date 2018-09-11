#!/usr/bin/env python
# encoding: utf-8

"""Pulse extraction using CHROM algorithm (%(version)s)

Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...]
           [--pulsedir=<path>]
           [--start=<int>] [--end=<int>] [--motion=<float>]
           [--threshold=<float>] [--skininit]
           [--framerate=<int>] [--order=<int>]
           [--window=<int>] [--gridcount]
           [--overwrite] [--verbose ...] [--plot]

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this screen
  -V, --version             Show version
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -o, --pulsedir=<path>     The path to the directory where signal extracted 
                            from the face area will be stored [default: pulse]
  --start=<int>             Starting frame index [default: 0].
  --end=<int>               End frame index [default: 0].
  --motion=<float>          The percentage of frames you want to select where the 
                            signal is "stable". 0 mean all the sequence [default: 0.0]. 
  --threshold=<float>       Threshold on the skin color probability [default: 0.5].
  --skininit                If you want to reinit the skin color distribution
                            at each frame.
  --framerate=<int>         Framerate of the video sequence [default: 61]
  --order=<int>             Order of the bandpass filter [default: 128]
  --window=<int>            Window size in the overlap-add procedure. A window
                            of zero means no procedure applied [default: 0].
  --gridcount               Tells the number of objects that will be processed.
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.


Example:

  To run the pulse extraction 

    $ %(prog)s config.py -v

See '%(prog)s --help' for more information.

"""
#from __future__ import print_function

import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt

import os
import sys
import pkg_resources

import numpy
import cv2
import dlib


#from utils import crop_face
from utils import build_bandpass_filter 

from extract_utils import compute_mean_rgb
from extract_utils import project_chrominance
from extract_utils import compute_gray_diff
from extract_utils import select_stable_frames 


from imutils.video import VideoStream
from imutils import face_utils
import imutils

import argparse




def main(user_input=None):

    basedir = './'

    # EXTRACT PULSE
    pulsedir = basedir + 'pulse'
    start = 0
    end = 450 #min 387
    motion = 0
    threshold = 0.1
    #skininit = True
    order = 128
    window = 20 #even
    framerate = 30

    # FREQUENCY ANALYSIS
    hrdir = basedir + 'hr'
    nsegments = 12
    nfft = 2048

    # RESULTS
    resultdir = basedir + 'results'
  
    overwrite = True
    plot =  True
    gridcount =  False
    verbosity_level =  0

    use_only_forehead = False
  
    # build the bandpass filter one and for all
    bandpass_filter = build_bandpass_filter(framerate, order, plot)

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help = "path to the (optional) video file")
    args = vars(ap.parse_args())


    if not args.get("video", False):
        from_webcam = True
        camera = cv2.VideoCapture(0)
        start = 0
        end = 450
	# otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])

    
    start_index = start
    end_index = end
   

    # number of final frames
    if end_index > 0:
        nb_frames = end_index - start_index

    # the grayscale difference between two consecutive frames (for stable frame selection)
    #if motion:
    #    diff_motion = numpy.zeros((nb_frames-1, 1),  dtype='float64')

    # skin color filter
    #skin_filter = bob.ip.skincolorfilter.SkinColorFilter()

    # output data
    output_data = numpy.zeros(nb_frames, dtype='float64')
    chrom = numpy.zeros((nb_frames, 2), dtype='float64')

    # loop on video frames
    frame_counter = 0
    i = start_index

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/pavi/Desktop/Machine_Learning/face_anti_spoofing/code/Bounded-Kalman-filter-method-for-motion-robust-non-contact-heart-rate-estimation/shape_predictor_68_face_landmarks.dat")

   
    while (i >= start_index and i < end_index):
        (grabbed, frame) = camera.read()
        print("Processing frame %d/%d...", i+1, end_index)

        if not grabbed:
            continue

        #if from_webcam:

        
        h,w,_ = frame.shape
        if h>w:
            ratio_forehead_height = 1.2
        elif w>h:
            ratio_forehead_height = 1.4
        else:
            ratio_forehead_height = 1.3  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects)==0:
            i += 1
            continue

        show_frame = frame.copy()
        ixc,iyc,a1x,a1y,a2x,a2y,a3x,a3y,a4x,a4y,a5lx,a5y,b1x,b1y,b2x,b2y,b3x,b3y,b4x,b4y,b5y,b5x,c1x,c1y,c2x,c2y,c3x,c3y,c4x,c4y,c5x,c5y,d1x,d1y,e1x,e1y=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        for rect in rects:
            #print(d.rect)
            print("Left: {} Top: {} Right: {} Bottom: {}".format(
                rect.left(), rect.top(), rect.right(), rect.bottom()))
            
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            counter = 0

            for (x, y) in shape:
                cv2.circle(show_frame, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(show_frame,str(counter),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1)
               
                #saving particular face landmarks for the ROI box
                if counter==21:
                    a1x=x
                    a1y=y/ratio_forehead_height ####
                if counter==22:
                    a2x=x
                    a2y=y
                if counter==27:
                    a3x=x
                    a3y=y
                if counter==8:
                    a4x=x
                    a4y=y
                if counter==23:
                    a5x=x
                    a5y=y
            
        
                if counter==17:
                    b1x=x
                    b1y=y*1.2
                if counter==31:
                    b2x=x
                    b2y=y
                if counter==28:
                    b3x=x 
                    b3y=y
                if counter==39:
                    b4x=x - 10 ###
                    b4y=y
                    ixc= (a1x+a2x)/2.2
                    iyc= (a4y + a3y)
            
                if counter==26:
                    c1x=x 
                    c1y=y/1.2
                if counter==35:
                    c2x=x
                    c2y=y
                if counter==28:
                    c3x=x
                    c3y=y
                if counter==42:
                    c4x=x + 10
                    c4y=y
        
        
                if counter==16:
                    d1x=x*1.1
                    d1y=y        
        
                if counter==0:
                    e1x=x/1.15
                    e1y=y

            
                counter+=1
                
                
            #co-ordinates for the rectangle 		
            listforehead = [int(a1x), int(a1y), a2x , a2y]
            listleftface = [int(b1x),int(b3y), b4x, b2y]
            listrightface = [int(c1x), int(c3y), c4x, c2y]
        
            
            left = rect.left()
            top =  rect.top()
            bottom = rect.bottom()
            right = rect.right()

            '''
            # motion difference (if asked for)
            if motion > 0 and (i < (nb_frames - 1)) and (counter > 0):
                current = frame[top:bottom,left:right]
                #current = crop_face(frame, bbox, bbox.size[1])
                diff_motion[counter-1] = compute_gray_diff(face, current)
            '''
            
            #face = crop_face(frame, bbox, bbox.size[1])
            face = frame[top:bottom,left:right]

            '''
            if plot and verbosity_level >= 2:
                from matplotlib import pyplot
                pyplot.imshow(numpy.rollaxis(numpy.rollaxis(face, 2),2))
                pyplot.show()
            
            # skin filter
            if counter == 0 or skininit:
                skin_filter.estimate_gaussian_parameters(face)
                logger.debug("Skin color parameters:\nmean\n{0}\ncovariance\n{1}".format(skin_filter.mean, skin_filter.covariance))
                skin_mask = skin_filter.get_skin_mask(face, threshold)
            

            if plot and verbosity_level >= 2:
                from matplotlib import pyplot
                skin_mask_image = numpy.copy(face)
                skin_mask_image[:, skin_mask] = 255
                pyplot.imshow(numpy.rollaxis(numpy.rollaxis(skin_mask_image, 2),2))
                pyplot.show()
            '''

            #compute mean
            forehead_roi = frame[listforehead[1]:listforehead[3],listforehead[0]:listforehead[2]]
            left_cheek_roi = frame[listleftface[1]:listleftface[3],listleftface[0]:listleftface[2]]
            right_cheek_roi = frame[listrightface[1]:listrightface[3],listrightface[2]:listrightface[0]]

            if use_only_forehead:
                r = numpy.mean(forehead_roi[:,:,2]) 
                g = numpy.mean(forehead_roi[:,:,1]) 
                b = numpy.mean(forehead_roi[:,:,0])
            else:
                r = numpy.mean(numpy.concatenate([forehead_roi[:,:,2].flatten(),left_cheek_roi[:,:,2].flatten(),right_cheek_roi[:,:,2].flatten()]))
                g = numpy.mean(numpy.concatenate([forehead_roi[:,:,1].flatten(),left_cheek_roi[:,:,1].flatten(),right_cheek_roi[:,:,1].flatten()]))
                b = numpy.mean(numpy.concatenate([forehead_roi[:,:,0].flatten(),left_cheek_roi[:,:,0].flatten(),right_cheek_roi[:,:,0].flatten()]))
            print(("Mean color -> R = {0}, G = {1}, B = {2}".format(r,g,b)))
            
            chrom[frame_counter] = project_chrominance(r, g, b)
            print("Chrominance -> X = {0}, Y = {1}".format(chrom[frame_counter][0], chrom[frame_counter][1]))

            
            '''
            # sometimes skin is not detected !
            if numpy.count_nonzero(skin_mask) != 0:        

                # compute the mean rgb values of the skin pixels
                r,g,b = compute_mean_rgb(face, skin_mask)
                logger.debug("Mean color -> R = {0}, G = {1}, B = {2}".format(r,g,b))

                # project onto the chrominance colorspace
                chrom[counter] = project_chrominance(r, g, b)
                logger.debug("Chrominance -> X = {0}, Y = {1}".format(chrom[counter][0], chrom[counter][1]))

            else:
                logger.warn("No skin pixels detected in frame {0}, using previous value".format(i))
                # very unlikely, but it could happened and messed up all experiments (averaging of scores ...)
                if counter == 0:
                    chrom[counter] = project_chrominance(128., 128., 128.)
                else:
                    chrom[counter] = chrom[counter-1]
            '''

        
        cv2.rectangle(show_frame, (listforehead[0], listforehead[1]), (listforehead[2], listforehead[3]), (255,0,0), 2)
        cv2.rectangle(show_frame, (listleftface[0], listleftface[1]), (listleftface[2], listleftface[3]), (255,0,0), 2)
        cv2.rectangle(show_frame, (listrightface[0], listrightface[1]), (listrightface[2], listrightface[3]), (255,0,0), 2)


        if h>w and h>640:
                dim = (int(640 * (w/h)),640)    
                show_frame = cv2.resize(show_frame, dim, interpolation = cv2.INTER_LINEAR)
        if w>h and w>640:
                dim = (640, int(640 * (h/w)))
                show_frame = cv2.resize(show_frame, dim, interpolation = cv2.INTER_LINEAR)
         
        cv2.imshow("frame",show_frame)
        cv2.waitKey(1)
        frame_counter +=1
        i += 1
    
    
    '''
    # select the most stable number of consecutive frames, if asked for
    if motion > 0:
        n_stable_frames_to_keep = int(motion * nb_frames)
        logger.info("Number of stable frames kept for motion -> {0}".format(n_stable_frames_to_keep))
        index = select_stable_frames(diff_motion, n_stable_frames_to_keep)
        logger.info("Stable segment -> {0} - {1}".format(index, index + n_stable_frames_to_keep))
        chrom = chrom[index:(index + n_stable_frames_to_keep),:]

    if plot:
        from matplotlib import pyplot
        f, axarr = pyplot.subplots(2, sharex=True)
        axarr[0].plot(range(chrom.shape[0]), chrom[:, 0], 'k')
        axarr[0].set_title("X value in the chrominance subspace")
        axarr[1].plot(range(chrom.shape[0]), chrom[:, 1], 'k')
        axarr[1].set_title("Y value in the chrominance subspace")
        pyplot.show()

    '''

    camera.release()
    cv2.destroyAllWindows()

    # now that we have the chrominance signals, apply bandpass
    from scipy.signal import filtfilt
    x_bandpassed = numpy.zeros(nb_frames, dtype='float64')
    y_bandpassed = numpy.zeros(nb_frames, dtype='float64')
    x_bandpassed = filtfilt(bandpass_filter, numpy.array([1]), chrom[:, 0])
    y_bandpassed = filtfilt(bandpass_filter, numpy.array([1]), chrom[:, 1])

    print("bandpassed")
    if plot:
        plt.rcParams['keymap.save'] = ''
        f, (axarr1,axarr2) = plt.subplots(2, sharex=True)
        print("plot")
        axarr1.plot(range(x_bandpassed.shape[0]), x_bandpassed, 'k')
        axarr1.set_title("X bandpassed")
        axarr2.plot(range(y_bandpassed.shape[0]), y_bandpassed, 'k')
        axarr2.set_title("Y bandpassed")
        plt.show()

    # build the final pulse signal
    alpha = numpy.std(x_bandpassed) / numpy.std(y_bandpassed)
    pulse = x_bandpassed - alpha * y_bandpassed
    print("pulse before Hanning window = ",pulse)

    # overlap-add if window_size != 0
    if window > 0:
        window_size = int(window)
        window_stride = int(window_size / 2)
        for w in range(0, (len(pulse)-window_size), window_stride):
            pulse[w:w+window_size] = 0.0
            xw = x_bandpassed[w:w+window_size]
            yw = y_bandpassed[w:w+window_size]
            alpha = numpy.std(xw) / numpy.std(yw)
            sw = xw - alpha * yw
            sw *= numpy.hanning(window_size)
            pulse[w:w+window_size] += sw
        
        if plot:
            from matplotlib import pyplot
            f, axarr = pyplot.subplots(1)
            pyplot.plot(range(pulse.shape[0]), pulse, 'k')
            pyplot.title("Pulse signal")
            pyplot.show()

    output_data = pulse
    print("pulse = ",pulse)
    signal = pulse

    
    if plot:
        from matplotlib import pyplot
        pyplot.plot(range(signal.shape[0]), signal, 'g')
        pyplot.title('Filtered green signal')
        pyplot.show()
    

    # find the segment length, such that we have 8 50% overlapping segments (Matlab's default)
    segment_length = (2*signal.shape[0]) // (nsegments + 1) 

    # the number of points for FFT should be larger than the segment length ...
    if nfft < segment_length:
        print("(nfft < nperseg): {0}, {1}".format(nfft,segment_length))
        

    from scipy.signal import welch
    green_f, green_psd = welch(signal, framerate, nperseg=segment_length, nfft=nfft)

    # find the max of the frequency spectrum in the range of interest
    first = numpy.where(green_f > 0.7)[0]
    last = numpy.where(green_f < 4)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)
    max_idx = numpy.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]
    hr = f_max*60.0
    print("Heart rate = {0}".format(hr))

    if plot:
        from matplotlib import pyplot
        pyplot.semilogy(green_f, green_psd, 'g')
        xmax, xmin, ymax, ymin = pyplot.axis()
        pyplot.vlines(green_f[range_of_interest[max_idx]], ymin, ymax, color='red')
        pyplot.title('Power spectrum of the green signal (HR = {0:.1f})'.format(hr))
        pyplot.show()
        
   

    #output_data = numpy.array([hr], dtype='float64')

    '''
    # saves the data into an HDF5 file with a '.hdf5' extension
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    bob.io.base.save(output_data, output)
    logger.info("Output file saved to `%s'...", output)
    '''



if __name__ == "__main__":
	main()
    
