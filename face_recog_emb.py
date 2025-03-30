import cv2
import numpy as np
import pickle as pkl
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from tensorflow import keras
from keras_facenet import FaceNet
from PIL import Image

# Function to get face embedding
def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    return model.predict(sample)[0]

# Load pre-trained models and encoders
with open("face_recog.pkl", "rb") as f:
    model = pkl.load(f)
with open("normalize.pkl", "rb") as f:
    norm = pkl.load(f)
with open("labelencoder.pkl", "rb") as f:
    le = pkl.load(f)

facenet = FaceNet()
embedder = facenet.model

def detect_faces(detector, cap, embedder , labels):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = cv2.flip(frame, 1)
        output = detector.detect_faces(img)
        
        for single in output:
            x, y, width, height = single["box"]
            x1, y1 = abs(x), abs(y)
            x2, y2 = x1 + width, y1 + height
            face = img[y1:y2, x1:x2]
            
            if face.size == 0:
                continue
            
            image = Image.fromarray(face).resize((160, 160))
            face_array = np.asarray(image)
            embedding = get_embedding(embedder, face_array)
            embedding = norm.transform(embedding.reshape(1, -1))
            
            try:
                pred = model.predict(embedding)
                pred_proba = model.predict_proba(embedding)
                confidence = f"Confidence: {np.max(pred_proba):.2f}"
                person = f"{labels[pred]}"
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            # Draw rectangle and text on frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(img, person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.putText(img, confidence, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    labels = le.inverse_transform([0, 1, 2, 3])
    detector = MTCNN()
    
    # Live camera feed 
    camera = cv2.VideoCapture(0)


    # reading from video
    video_path = "path"
    video = cv2.VideoCapture(video_path)
    
    if not camera.isOpened():
        print("Error loading video")
    else:
        detect_faces(detector, camera, embedder , labels)
