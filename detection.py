import cv2
import numpy as np
haar = cv2.CascadeClassifier('haarcascade_fullbody.xml')
video = cv2.VideoCapture('people.mp4')
e,frame = video.read()
(height,width) = frame.shape[:2]
def scale(height,width,scale_percent):
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    return dim
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output2.mp4',fourcc, 10,scale(height,width,100))
position = []
def onMouse(event,x,y,flags,param):
    global position
    if event == cv2.EVENT_LBUTTONDOWN:
        position.append([x,y])
while True:
    e,frame = video.read()
    if e:
        frame = cv2.resize(frame,scale(height,width,50),interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        detect = haar.detectMultiScale(gray)
        key = cv2.waitKey(1)
        if key == ord('p'):  # pause frame till any other key is pressed
            cv2.namedWindow('frame')
            cv2.setMouseCallback('frame',onMouse)
            cv2.waitKey(-1)
        position_array = np.array(position)
        position_array = position_array.reshape((-1,1,2))
        
        cv2.polylines(frame,[position_array],True,(0,255,255),2)
        for (x,y,w,h) in detect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            centre = (int(x+w/2),int(y+h/2))
            if (position_array.size != 0): # to check if point is inside region
                result = cv2.pointPolygonTest(position_array, centre, False)
                if(result == 1)or(result == 0):
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    
        cv2.imshow('frame',frame)
        frame = cv2.resize(frame,scale(height,width,100),interpolation = cv2.INTER_CUBIC)
        output.write(frame)
        if key == ord('q'):  # quit all frames
            break
video.release()
output.release()
cv2.destroyAllWindows()
