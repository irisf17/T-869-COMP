import numpy as np
import cv2
from collections import defaultdict
import sys

# I used Code from stackoverflow to help with this assignment. :)
# https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """
    Group lines by their angle using k-means clustering.

    Code from here:
    https://stackoverflow.com/a/46572063/1755401
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Get angles in [0, pi] radians
    angles = np.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)


    # python 3.x, syntax has changed.
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

    labels = labels.reshape(-1) # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())
    # print("Segmented lines into two groups: %d, %d" % (len(segmented[0]), len(segmented[1])))

    return segmented


def intersection(line1, line2):
    """
    Find the intersection of two lines 
    specified in Hesse normal form.

    Returns closest integer pixel locations.

    See here:
    https://stackoverflow.com/a/383527/5087436
    """

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [[x0, y0]]


def segmented_intersections(lines):
    """
    Find the intersection between groups of lines.
    """

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections


def drawLines(img, lines, color=(0,0,255)):
    """
    Draw lines on an image
    """
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1,y1), (x2,y2), color, 1)

cap = cv2.VideoCapture(1)

while(True):
    ret, frame = cap.read()
    edges = cv2.Canny(frame,50,200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect lines
    rho = 1
    theta = np.pi/90
    thresh = 100

    lines = cv2.HoughLines(edges, rho, theta, thresh)
    
    if ((lines is not None) and (len(lines) > 2)):
        # print("Found lines: %d" % (len(lines)))

        # Draw all Hough lines in red
        img_with_all_lines = np.copy(frame)
        drawLines(img_with_all_lines, lines)
        
        # Cluster line angles into 2 groups (vertical and horizontal)
        segmented = segment_by_angle_kmeans(lines, 2)

        # Find the intersections of each vertical line with each horizontal line
        intersections = segmented_intersections(segmented)
        # print(len(intersections))

        img_with_segmented_lines = np.copy(frame)

        # Draw vertical lines in green
        vertical_lines = segmented[1]
        img_with_vertical_lines = np.copy(frame)
        drawLines(img_with_segmented_lines, vertical_lines, (0,255,0))

        # Draw horizontal lines in yellow
        horizontal_lines = segmented[0]
        img_with_horizontal_lines = np.copy(frame)
        drawLines(img_with_segmented_lines, horizontal_lines, (0,255,255))

        # Draw intersection points in magenta
        counter = 0
        to_warp_points = []
        intersect_flag = False
        for point in intersections:
            # print(f"the length of intersections: {len(intersections)}")
            counter += 1
            # print(point)

            pt = (point[0][0], point[0][1])
            if len(intersections) == 4 and intersect_flag == False:
                intersect_flag = True
                for i in range(len(intersections)):
                    to_warp_points.append(intersections[i])
                    # print("ABBA")
                # print(to_warp_points)
            # print(to_warp_points)
            
            # print(to_warp_points)
            length = 5
            cv2.line(img_with_segmented_lines, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 1) # vertical line
            cv2.line(img_with_segmented_lines, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 1)

        if(intersect_flag == True):

            warped = np.copy(frame)
            # warped = cv2.resize(warped, 640,480)
            to_warp_points = np.array([x[0] for x in to_warp_points])
            after_warp_points =  np.float32([[0,0],[639,0],[0, 479],[639,479]])
            print('warp points; ', to_warp_points)
            print('dest points: ', after_warp_points)
            M, status = cv2.findHomography(to_warp_points,after_warp_points)
            # M = cv2.getPerspectiveTransform(to_warp_points,after_warp_points)
            dst = cv2.warpPerspective(warped,M,(640,480))
            cv2.imshow("frame warped", dst)        

        # cv2.imshow("original frame", frame)
        # cv2.imshow("canny", edges)
        # cv2.imshow("Hough lines", img_with_all_lines)
        # cv2.imshow("vert lines", img_with_vertical_lines)
        # cv2.imshow("hor lines", img_with_horizontal_lines)
        cv2.imshow("seg lines", img_with_segmented_lines)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()