# BGR standard fyrir video
# find the bright spot and mark it, function in opencv that finds the maximum value in the image, then use your own code to find the max value/brightest spot
# rgb and find the reddest spot

import cv2
import numpy as np
import time

# 0 for the computer camera, 1 for the phone with ip-number
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0


# finding the best BGR value for the color red
def hsv_color(red):
    red = np.uint8([[[0,0,255]]])
    hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
    print(hsv_red) 
    # [[[  0 255 255]]]

def bright_loc(gray):
    highest_val = 0
    highest_loc = (0,0)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i,j] > highest_val:
                highest_val = gray[i,j]
                highest_loc = (j, i)
    return highest_loc

def reddest_loc(hsv):
    highest_loc_red = (0,0)
    reddest_val = [0,100,100]

    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            if (hsv[i,j][0] == 0 and hsv[i,j][1] >= reddest_val[1] and hsv[i,j][2] >= reddest_val[2]):
                reddest_val = hsv[i, j]
                highest_loc_red = (j, i)
    return highest_loc_red

def read_im():
    image = cv2.imread('C:\\Users\\irisf\\Documents\\HR-master\\ComputerVision\\assignment_1\\mynd.jpg')

    cv2.imshow('img', image)

while(True):
    # read_im()
    start_time = time.time()

    ret, frame = cap.read()

    # --------- BRIGHTEST POINT using inbuilt function -------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #(480, 640) 2d
     # finding the brightest value with inbuilt function
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(frame, maxLoc, 5, (0, 0, 0), 2)
    
    # --------- finding brightest value with for loops ----------
    # brightest_loc = bright_loc(gray)
    # cv2.circle(frame, brightest_loc, 5, (0, 0, 0), 2)

    # ----------- REDDEST POINT using inbuilt function -----------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV
    lower_red = np.array([0,0,100]) #hue, saturation, intensity
    upper_red = np.array([0,255,255]) # BGR
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # finding max red value
    (minVal_red, maxVal_red, minLoc_red, maxLoc_red) = cv2.minMaxLoc(hsv[:,:,1],mask)
    cv2.circle(frame, maxLoc_red, 5, (0, 0, 255), 2)
    
    # # ---------- finding most reddest sport using FOR loops --------------
    # brightest_loc_red = reddest_loc(hsv)
    # cv2.circle(frame, brightest_loc_red, 5, (0, 0, 255), 2)

    # ------------ font to display FPS ------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for the frame
    new_frame_time = time.time()
 
    # Calculating the fps
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
    
    # putting the FPS count on the frame
    cv2.putText(frame, "FPS:" + str(fps), (1, 450), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Displaying image
    cv2.imshow('frame',frame)
    end_time = time.time()

    # print(end_time - start_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()