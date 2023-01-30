import cv2

#PRE-TRAINED MODELS
#configuration and weight of the model
net = cv2.dnn.readNet(r"object-detection-opencv/dnn_model-220107-114215/dnn_model/yolov4-tiny.weights",r"object-detection-opencv/dnn_model-220107-114215/dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
#size is adjusted cause high pixaled image is harder to process.size must adjusted when good detection and speed are maximum(they are inversly propotional)
model.setInputParams(size=(320,320), scale=1/255)

#load class list
classes=[]
t = open(r"object-detection-opencv/dnn_model-220107-114215/dnn_model/classes.txt") 
for i in t.readlines():
    c = i.strip()
    classes.append(c)
t.close()
    

#CAMERA
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#all the frames are needed(video) in real-time we loop the code for getting the current frame
#ret is a boolean variable that returns true if frame is available
while True:
    ret, frame =cap.read()

    #OBJECT DETECTION
    (class_ids,score,bboxes) = model.detect(frame) #detect the object on the frame
    for class_ids, score , bbox in zip(class_ids,score,bboxes):#zip funtion joins all the arrays into one array
        x, y, w ,h = bbox
        #x,y is the top left point
        #x+w,y+h is the bottom right point
        #enter bgr and thickness after the above two
        cv2.putText(frame,str(classes[class_ids]),(x,y - 10),cv2.FONT_HERSHEY_PLAIN,2,(50,0,200),2)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(50,0,200),3)


    
    cv2.imshow("Frame",frame)
    #if waitKey(0) is used it waits for the user to press a key to move to the next frame
    #using waitKey(1) is waiting 1ms before the next frame
    cv2.waitKey(1)