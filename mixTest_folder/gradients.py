import numpy as np
import cv2

# cap = cv2.VideoCapture(0)

while(1):
    # ret, frame = cap.read()
    img = cv2.imread("C:\\Users\\irisf\\Documents\\HR-master\\ComputerVision\\assignment_3\\mynd.jpg")
    img = cv2.resize(img, (640, 480))

    # img = cv2.imread('dave.jpg',0)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    cv2.imshow("FRAME", img)
    cv2.imshow("laplace", laplacian)
    cv2.imshow("sobelx", sobelx)
    cv2.imshow("sobely", sobely)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()