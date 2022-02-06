import face_recognition
import pickle
import cv2
import os
import time

def parse_known_faces(path):
    faces_path = os.path.join(path, "known_faces")
    knownEncodings = []
    names = []

    for root, dirs, files in os.walk(faces_path, topdown=False):
        for fname in files:
            file_path = os.path.join(root, fname)
            print(file_path)
            # break
            name = file_path.split(os.path.sep)[-2]
            cap = cv2.VideoCapture(file_path)
            
            count = 0
            while True:
                success, img = cap.read()
                if not success:
                    break
                count += 1
                if count % 15 != 0:
                    continue
                rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                start = time.time()
                boxes = face_recognition.face_locations(rgb, model = 'cnn')
                if len(boxes) == 0:
                    continue 
                encodings = face_recognition.face_encodings(rgb, boxes)
                print(time.time()-start)

                for encoding in encodings:
                    knownEncodings.append(encoding)
                    names.append(name)
            cap.release()

    data = {'encodings':knownEncodings,"names":names}
    encoding_file = os.path.join(path, "known_faces_processed/encodings")
    with open(encoding_file,'wb') as f:
        f.write(pickle.dumps(data))

def main():
    path = '.'
    parse_known_faces(path)

if __name__ == '__main__':
    main()

