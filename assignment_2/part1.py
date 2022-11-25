import cv2
import sys
import math
import numpy as np
import time

cap = cv2.VideoCapture(0)

# finna tvo random pkt út frá canny edges
# finn linu í gegnum þessa tvo punkta með numpy
# krossfelda og finna distance-ið, telja inliers, decide distance-ið, margin lengdina


def random_points(edges):
    random_p1 = (0,0)
    random_p2 = (0,0)

    points = np.argwhere(edges != 0)
    flag = True

    if (len(points) == 0):
        flag = False
        return points, random_p1, random_p2, flag

    rand_1 = np.random.randint(0,(len(points))) 
    random_p1 = points[rand_1]
    rand_2 = np.random.randint(0,(len(points))) 
    random_p2 = points[rand_2]

    return points, random_p1, random_p2, flag


while(True):
    ret, frame = cap.read()

    start_time = time.time()

    # cv2.findcountour()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # dst = cv2.medianBlur(frame,27)  
    
    edges = cv2.Canny(frame,50,200, apertureSize = 3)
    # Finding best points
    max_inliers = 0

    for k in range(30):
        edge_points, point1, point2, flag = random_points(edges)
        if(flag == False):
            break
        inlier = 0
        for j in range(0, len(edge_points), 20):
            p = edge_points[j]
            dist = np.linalg.norm(np.cross(point2-point1, point1-p))/np.linalg.norm(point2-point1)
            if(dist < 4):
                inlier += 1
            # store which point pair has the most inliers
            if (inlier > max_inliers):
                max_inliers = inlier
                best_point1 = point1
                best_point2 = point2

    x_cords = np.empty(max_inliers)
    y_cords = np.empty(max_inliers)
    index = 0
    # STORE COORDINATES OF THOSE INLIERS
    for i in range(0, len(edge_points), 20):
        p = edge_points[i]
        d = np.linalg.norm(np.cross(best_point2-best_point1, best_point1 - p))/np.linalg.norm(best_point2-best_point1)
        if(d < 4):
            x_cords[index] = p[0]
            y_cords[index] = p[1]
            index += 1
    # FIND LINE EQUATION
    if (x_cords != []):
        h, x = np.polyfit(x_cords, y_cords, 1) #for 1st degree equation
    else:
        h = 1

    y1 = int(-1000*h + x)
    y2 = int(1000*h + x)
    # PLOT
    cv2.line(frame, (y1, -1000), (y2, 1000), (0, 255, 0), 2)
    frame = cv2.resize(frame, (640, 480))

    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    cv2.imshow("FRAME", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()