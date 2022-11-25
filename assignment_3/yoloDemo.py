import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'C:\\Users\\irisf\\Documents\\HR-master\\ComputerVision\\assignment_3\\coco.names'
classNames = []

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))

# modelConfiguration = 'C:\\Users\\irisf\\Documents\\HR-master\\ComputerVision\\assignment_3\\yolov3-320.cfg'
# modelWeights = 'C:\\Users\\irisf\\Documents\\HR-master\\ComputerVision\\assignment_3\\yolov3.weights'
modelConfiguration = 'C:\\Users\\irisf\\Documents\\HR-master\\ComputerVision\\assignment_3\\yolov3-tiny.cfg'
modelWeights = 'C:\\Users\\irisf\\Documents\\HR-master\\ComputerVision\\assignment_3\\yolov3-tiny.weights'

# creating the network
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs,img):
    hT, wT, cT = img.shape
    # store our values
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            # index of the maximum value = max confidence value
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    # eliminates overlapping boxes, based on confidence value, picks the highest confidence
    # indices the boxes to keep
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    # print(indices)
    for i in indices:
        # eliminate the brackets
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        # draw the box
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        # draw the text of item and text of confidence
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)



while True:
    success, img = cap.read()
    startime = time.time()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    # input to our network
    net.setInput(blob)

    # to get out names of layers, 3 different outputs
    layerNames = net.getLayerNames()
    # print(layerNames)
    # print(net.getUnconnectedOutLayers())
    # output names of our layers
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)
    # print(len(outputs))
    # print(outputs[0].shape) #300,85
    # print(outputs[1].shape) # 1200, 85
    # print(outputs[2].shape) # 4800, 85

    
    findObjects(outputs,img)
    endtime = time.time()
    total_time = endtime - startime
    print(total_time)

    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break