# Face Recognition System

## Overview

This face recognition system is designed to detect and recognize faces using a combination of MTCNN, FaceNet, and SVM. The system extracts facial features and performs classification to identify individuals. 
It has the accuracy of about 95% on real time feed with just 20 images of each class. 

## Workflow

Face Detection: Uses MTCNN (Multi-Task Cascaded Convolutional Networks) to detect faces in an image.

Feature Extraction: Extracts face embeddings using FaceNet, which maps faces into a high-dimensional space.

Face Classification: Uses an SVM (Support Vector Machine) classifier to recognize faces based on the extracted embeddings.

## Source Code 

Soon I will add the jupyter notebook for the same 

## Applications

Attendance systems<br>
Security and surveillance<br>
User authentication

## Future Improvements

Adding Anti-Spoof filter  
Improve accuracy with fine-tuning
Implement real-time face recognition using a webcam
Extend support for multiple faces in an image

## License

This project is open-source and available for further improvements.
