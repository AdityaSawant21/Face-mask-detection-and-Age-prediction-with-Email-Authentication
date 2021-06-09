# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from tkinter import*
import cv2
from tkinter import messagebox
import smtplib

# ---------------------------------------------------------
import argparse
import time
# ----------------------------------------------------

from keras.preprocessing.image import img_to_array
from email.message import EmailMessage


def detect_and_predict_mask(frame, faceNet, MaskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = MaskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# --------------------------------------------------------------------------------


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


# ------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

face_cascade_Path = "haarcascade_frontalface_default.xml"


faceCascade = cv2.CascadeClassifier(face_cascade_Path)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
# names related to ids: The names associated to the ids: 1 for Mohamed, 2 for Jack, etc...
names = ['None', 'Barun', 'Nikhil', 'Aditya',
         'Tayde', 'Sawant']  # add a name into this list
# --------------------

# load our serialized face detector model from disk
prototxtPath = r"E:\app\Face Mask Detection and Alert System\face_detector\deploy.prototxt"
weightsPath = r"E:\app\Face Mask Detection and Alert System\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
MaskNet = load_model("mask_detector.model")


parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-3)', '(4-7)', '(8-14)', '(15-20)',
           '(21-32)', '(33-43)', '(44-53)', '(54-100)']  # 8 Age Groups
genderList = ['Male', 'Female']

padding = 20
t = 0
faceNet1 = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
# ----------------------------------------------------------------------

# initialize the video stream
print("Starting video stream...")
cam = cv2.VideoCapture(0)  # Starts VideoStream
vs = VideoStream(0)
cam.set(3, 250)
cam.set(4, 250)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    start_point = (15, 15)
    end_point = (370, 80)
    thickness = -1
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=400)
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, MaskNet)

 # -------------------------------------------------------------------------------
    resultImg, faceBoxes = highlightFace(faceNet1, frame)
 # -------------------------------------------------------------------------------
    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutmask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text

        label = "Mask" if mask > withoutmask else "No Mask"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)

        # display the label and bounding box rectangle on the output

        # frame

        cv2.putText(frame, label, (startX, startY - 10),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        label = "Mask" if mask > withoutmask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
 # -----------------
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                # Unknown Face
                id = "Who are you ?"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5),
                        font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5),
                        font, 1, (255, 255, 0), 1)

        cv2.imshow('Camera', img)
 # -------------
        # include the probability in the label
        if(label == 'No Mask'):
            t = t + 1
            time.sleep(1)
            print(t)
            if(t == 5 or t == 12 or t == 18):
                messagebox.showwarning("Warning", "Please wear a Face Mask")
            if(t == 20):

                cv2.imwrite("./Output/detected.jpg", frame)
                messagebox.showwarning(
                    "Warning", "Access Denied. Please wear a Face Mask")

                msg = EmailMessage()
                msg['Subject'] = 'Subject - Attention!! Someone violated our facemask policy.'
                # Write Sender's email
                msg['From'] = 'Senders email'
                # Write Reciever's email(Authority Email)
                msg['To'] = 'Authority Email'
                msg.set_content(
                    'Respected Authority,\n         Some Person has been detected without a face mask. Below is the attached image of that person.')

                with open("Output/detected.jpg", "rb") as f:
                    fdata = f.read()
                    fname = f.name
                    msg.add_attachment(fdata, maintype='Image',
                                       subtype="jpg", filename=fname)

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    # Write Sender's email and password
                    smtp.login('Senders email', 'Senders Password')
                    smtp.send_message(msg)
                print('Alert mail Sent to authorities')
        elif(label == 'Mask'):
            pass
            break
        else:
            print("Invalid")
        print("Saving image...")
        # detected.jpg file will be created
        cv2.imwrite("./Output/detected.jpg", frame)

        label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

 # show the output frame
        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

 # ----------------------------------------------------------------------------------
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):
                     min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        blob1 = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob1)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob1)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (
            faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Detect Age & Gender", resultImg)

cv2.destroyAllWindows()
vs.stop()

# autopep8 -i detect_mask_video.py
# python detect_mask_video.py
