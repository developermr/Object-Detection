import cv2
import numpy as np
from gtts import gTTS
import os


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")#training models of yolov3
classes = []
with open("namefiles.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("images/p1.jpg")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)
img = cv2.resize(img, None, fx=0.8, fy=0.8)# output screen size

height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []#namefiles is use for matching with id
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)# detect object width
            center_y = int(detection[1] * height)# detect object height
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

print(indexes)
font = cv2.FONT_HERSHEY_DUPLEX #set the label font style
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x - 10, y - 10), (x +w , y+h), color, 2)#make rectange arround objects
        cv2.putText(img, label, (x, y - 20), font, 1, color, 2)# put label with crrosponding pictures


mytext = label
language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=True)
myobj.save("welcome.mp3")
os.system("start welcome.mp3")



cv2.imshow("Image", img)#show the output

cv2.waitKey(0)
cv2.destroyAllWindows()
