import face_recognition
import pickle
import cv2
import os
import time
import numpy as np
from tensorflow.keras.models import load_model


def transform_image(img):
    face = cv2.resize(img,(32,32))
    face = face.astype("float") / 255.0
    face = face[np.newaxis,...]
    return face



def start_face_recognition(vid_path = None, path = '.'):
    encoding_path = os.path.join(path, 'known_faces_processed/encodings')
    data = pickle.loads(open(encoding_path,'rb').read())
    # Liveliness model

    output_path = os.path.splitext(vid_path)[0] + "_output.mp4"
    model_path = os.path.join(path,'liveliness/model/liveliness.h5')
    labels_path = os.path.join(path,'liveliness/model/labels/labels')
    model = load_model(model_path)
    le = pickle.loads(open(labels_path, "rb").read())
  
    if vid_path == None:
        cap = cv2.VideoCapture(0)
    else :
        cap = cv2.VideoCapture(vid_path)

    writer = None
    while True:
        # starting to read
        success, img = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model = 'cnn')
        # print(boxes)
        # continue
        if len(boxes) == 0:
            continue
        
        fin_boxes = []
        for box in boxes:
            x1,y1,x2,y2 = box[3],box[0],box[1],box[2]
            face = img[y1:y2,x1:x2]
            face = transform_image(face)
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
            if 'real' in label:
              fin_boxes.append(box)
            else :
              cv2.putText(img,'Fake',(x1,y1-20),cv2.FONT_HERSHEY_SIMPLEX,
              1,(255,0,0),1,2)
              cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0),2)

        encodings = face_recognition.face_encodings(rgb, fin_boxes)
        for encoding, box in zip(encodings, fin_boxes):
            matches = face_recognition.compare_faces(data['encodings'], encoding)
            name = 'UNKNOWN'
            if True in matches:
              idx = [i for i,b in enumerate(matches) if b]
              counts = {}
              for i in idx:
                name = data['names'][i]
                counts[name] = counts.get(name,0) + 1
              name = max(counts, key = counts.get)
            x1,y1,x2,y2 = box[3],box[0],box[1],box[2]
            cv2.putText(img,name,(x1,y1-20),cv2.FONT_HERSHEY_SIMPLEX,
              1,(0,255,0),1,2)
            cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0),2)
        
        if writer is None:
          fourcc = cv2.VideoWriter_fourcc(*"MJPG")
          writer = cv2.VideoWriter(output_path, fourcc, 20,
            (img.shape[1], img.shape[0]), True)
        else:
          writer.write(img)


def main():
    pass

if __name__ == '__main__':
    main()