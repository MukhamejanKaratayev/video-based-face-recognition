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

    output_path = os.path.splitext(vid_path)[0] + "_output_new.mp4"
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
        
        labels = []
        for box in boxes:
            x1,y1,x2,y2 = box[3],box[0],box[1],box[2]
            face = img[y1:y2,x1:x2]
            face = transform_image(face)
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
            
            if 'real' in label:
              labels.append(True)
            else :
              labels.append(False)
              # cv2.putText(img,'Fake',(x1,y1-20),cv2.FONT_HERSHEY_SIMPLEX,
              # 1,(0,0,255),1,2)
              # cv2.rectangle(img, (x1,y1),(x2,y2), (0,0,255),2)

        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding, box, label in zip(encodings, boxes, labels):
            matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance = 0.4)
            name = 'UNKNOWN'
            color = (0,255,0)
            rec_color = (0,255,0)
            if True in matches:
              idx = [i for i,b in enumerate(matches) if b]
              counts = {}
              for i in idx:
                name = data['names'][i]
                counts[name] = counts.get(name,0) + 1
              name = max(counts, key = counts.get)
            x1,y1,x2,y2 = box[3],box[0],box[1],box[2]
            if name == 'UNKNOWN':
              color = (0,0,255)
            if not label:
              rec_color = (0,0,255)
            cv2.putText(img,name,(x1,y1-20),cv2.FONT_HERSHEY_SIMPLEX,
              1,color,1,2)
            cv2.rectangle(img, (x1,y1),(x2,y2), rec_color,2)
        
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