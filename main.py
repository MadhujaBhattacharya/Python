import cv2 
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn= frame.copy()
    frameHeight= frameOpencvDnn.shape[0]
    frameWidth= frameOpencvDnn.shape[1]
    blob= cv2.dnn.blobFromImage(frameOpencvDnn, 1.1,(300,300),[104,117,123],True,False)
    net.setInput(blob)
    detections= net.forward()
    faceBoxes=[]

    for i in range(detections.shape[2]):
        confidence= detections[0,0,i,2]
        if confidence > conf_threshold:
            x1= int(detections[0,0,i,3]*frameWidth)
            y1= int(detections[0,0,i,4]*frameHeight)
            x2= int(detections[0,0,i,5]*frameWidth)
            y2= int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn,(x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)),8)
    return frameOpencvDnn, faceBoxes



parser= argparse.ArgumentParser()
parser.add_argument('--image')
args= parser.parse_args()

FaceProto= "opencv_face_detector.pbtxt"
FaceModel= "opencv_face_detector_uint8.pb"
AgeProto= "age_deploy.prototxt"
AgeModel= "age_net.caffemodel"
GenderProto= "gender_deploy.prototxt"
GenderModel= "gender_net.caffemodel"
MODEL_MEAN_VALUES= (78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

FaceNet= cv2.dnn.readNet(FaceModel, FaceProto)
AgeNet= cv2.dnn.readNet(AgeModel, AgeProto)
GenderNet= cv2.dnn.readNet(GenderModel, GenderProto)

video= cv2.VideoCapture(args.image if args.image else 0)
padding= 20 

while cv2.waitKey(1)<0:
    hasFrame, frame= video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    resultImage, faceBoxes= highlightFace(FaceNet, frame)
    if not faceBoxes:
        print("No Face Detected")
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        GenderNet.setInput(blob)
        genderPreds=GenderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        AgeNet.setInput(blob)
        agePreds=AgeNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImage, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImage)




