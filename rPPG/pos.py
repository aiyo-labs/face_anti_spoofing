"""
Pulse extraction using POS algorithm (%(version)s)
"""


import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt

import os
import sys
import pkg_resources

import numpy as np
import cv2
import dlib


#from utils import crop_face
from utils import build_bandpass_filter 

from extract_utils import project_chrominance
from extract_utils import compute_gray_diff
from extract_utils import select_stable_frames 


from imutils.video import VideoStream
from imutils import face_utils
import imutils

import argparse
import os


def main(user_input=None):

    

    # EXTRACT PULSE
    pulsedir ="/Volumes/MacMini-Backups/siw-db/live/pulse/"
    start = 0
    end = 300
    motion = 0
    threshold = 0.1
    #skininit = True
    order = 128
    framerate = 30

    # FREQUENCY ANALYSIS
    nsegments = 12
    nfft = 2048

    overwrite = True
    plot =  True
    gridcount =  False
    verbosity_level =  0

    use_only_forehead = True
  
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

    video_file_path = args["video"]
    video_file_name = os.path.basename(video_file_path)
    
    start_index = start
    end_index = end

    # number of final frames
    if end_index > 0:
        nb_frames = end_index - start_index


    # output data
    output_data = np.zeros(nb_frames, dtype='float64')
    chrom = np.zeros((nb_frames, 2), dtype='float64')

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
                    a1y = abs(rect.bottom()-((rect.bottom()- rect.top()) * 1.04))
                    #a1y=y/ratio_forehead_height ####
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

            #face = crop_face(frame, bbox, bbox.size[1])
            face = frame[top:bottom,left:right]

            #compute mean
            forehead_roi = frame[listforehead[1]:listforehead[3],listforehead[0]:listforehead[2]]
            left_cheek_roi = frame[listleftface[1]:listleftface[3],listleftface[0]:listleftface[2]]
            right_cheek_roi = frame[listrightface[1]:listrightface[3],listrightface[2]:listrightface[0]]

            if use_only_forehead:
                r = np.mean(forehead_roi[:,:,2]) 
                g = np.mean(forehead_roi[:,:,1]) 
                b = np.mean(forehead_roi[:,:,0])
            else:
                r = np.mean(np.concatenate([forehead_roi[:,:,2].flatten(),left_cheek_roi[:,:,2].flatten(),right_cheek_roi[:,:,2].flatten()]))
                g = np.mean(np.concatenate([forehead_roi[:,:,1].flatten(),left_cheek_roi[:,:,1].flatten(),right_cheek_roi[:,:,1].flatten()]))
                b = np.mean(np.concatenate([forehead_roi[:,:,0].flatten(),left_cheek_roi[:,:,0].flatten(),right_cheek_roi[:,:,0].flatten()]))
            
            if frame_counter==0:
                mean_rgb = np.array([r,g,b])
            else:
                mean_rgb = np.vstack((mean_rgb,np.array([r,g,b])))

            #chrom[frame_counter] = project_chrominance(r, g, b)
            print("Mean RGB -> R = {0}, G = {1}, B = {2} ".format(r,g,b))

        
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
        #end loop
    
    camera.release()
    cv2.destroyAllWindows()

    if plot:
        f = np.arange(0,mean_rgb.shape[0])
        plt.plot(f, mean_rgb[:,0] , 'r', f,  mean_rgb[:,1], 'g', f,  mean_rgb[:,2], 'b')
        plt.title("Mean RGB - Complete")
        plt.show()

    #Calculating l
    l = int(framerate * 1.6)
    print("Framerate : ",l)

    H = np.zeros(mean_rgb.shape[0])

    for t in range(0, (mean_rgb.shape[0]-l)):
        #t = 0
        # Step 1: Spatial averaging
        C = mean_rgb[t:t+l-1,:].T
        #C = mean_rgb.T
        print("C shape", C.shape)
        print("t={0},t+l={1}".format(t,t+l))
        if t == 3:
            plot = False

        if plot:
            f = np.arange(0,C.shape[1])
            plt.plot(f, C[0,:] , 'r', f,  C[1,:], 'g', f,  C[2,:], 'b')
            plt.title("Mean RGB - Sliding Window")
            plt.show()
        
        #Step 2 : Temporal normalization
        mean_color = np.mean(C, axis=1)
        #print("Mean color", mean_color)
        
        diag_mean_color = np.diag(mean_color)
        #print("Diagonal",diag_mean_color)
        
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        #print("Inverse",diag_mean_color_inv)
        
        Cn = np.matmul(diag_mean_color_inv,C)
        #Cn = diag_mean_color_inv@C
        #print("Temporal normalization", Cn)
        #print("Cn shape", Cn.shape)

        if plot:
            f = np.arange(0,Cn.shape[1])
            #plt.ylim(0,100000)
            plt.plot(f, Cn[0,:] , 'r', f,  Cn[1,:], 'g', f,  Cn[2,:], 'b')
            plt.title("Temporal normalization - Sliding Window")
            plt.show()
    
        #Step 3: 
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(projection_matrix,Cn)
        #S = projection_matrix@Cn
        print("S matrix",S)
        print("S shape", S.shape)
        if plot:
            f = np.arange(0,S.shape[1])
            #plt.ylim(0,100000)
            plt.plot(f, S[0,:] , 'c', f,  S[1,:], 'm')
            plt.title("Projection matrix")
            plt.show()

        #Step 4:
        #2D signal to 1D signal
        std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
        print("std",std)
        P = np.matmul(std,S)
        #P = std@S
        print("P",P)
        if plot:
            f = np.arange(0,len(P))
            plt.plot(f, P, 'k')
            plt.title("Alpha tuning")
            plt.show()

        #Step 5: Overlap-Adding
        H[t:t+l-1] = H[t:t+l-1] +  (P-np.mean(P))/np.std(P)

    print("Pulse",H)
    signal = H
    print("Pulse shape", H.shape)
    plot = True

    #FFT to find the maxiumum frequency
    # find the segment length, such that we have 8 50% overlapping segments (Matlab's default)
    segment_length = (2*signal.shape[0]) // (nsegments + 1) 

    # the number of points for FFT should be larger than the segment length ...
    if nfft < segment_length:
        print("(nfft < nperseg): {0}, {1}".format(nfft,segment_length))
        
    print("nperseg",segment_length)
    
    if plot:
        from matplotlib import pyplot
        pyplot.plot(range(signal.shape[0]), signal, 'g')
        pyplot.title('Filtered green signal')
        pyplot.show()

    

    from scipy.signal import welch
    signal = signal.flatten()
    green_f, green_psd = welch(signal, framerate, 'flattop', nperseg=segment_length) #, scaling='spectrum',nfft=2048)
    print("Green F, Shape",green_f,green_f.shape)
    print("Green PSD, Shape",green_psd,green_psd.shape)

    #green_psd = green_psd.flatten()
    first = np.where(green_f > 0.9)[0]
    last = np.where(green_f < 1.8)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)

    print("Range of interest",range_of_interest)
    max_idx = np.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]

    hr = f_max*60.0
    print("Heart rate = {0}".format(hr))

    import scipy.io as sio
    mat_file_name = pulsedir + "pulse_" + video_file_name[:-4] + "_frame-0-15" + ".mat"
    sio.savemat(mat_file_name,{'pulse':signal, 'heartrate':hr, 'nperseg':segment_length})


    if plot:
        from matplotlib import pyplot
        pyplot.semilogy(green_f, green_psd, 'g')
        xmax, xmin, ymax, ymin = pyplot.axis()
        pyplot.vlines(green_f[range_of_interest[max_idx]], ymin, ymax, color='red')
        pyplot.title('Power spectrum of the green signal (HR = {0:.1f})'.format(hr))
        pyplot.show()
        

if __name__ == "__main__":
	main()
    