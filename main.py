import os
import face_recognition
import cv2
import numpy as np
from flask_socketio import SocketIO
from flask import Flask, render_template, Response 
import socket  # for getting the hostname and server address of the device running the script
import time
import threading
import functools

#host and server address
hostname = socket.gethostname()
server_Add = socket.gethostbyname(hostname)

appRecognition = Flask(__name__)

sockApp = SocketIO(appRecognition)

#server address to console
print("http://" + server_Add + ":8080")

# Load a sample picture and learn how to recognize it.
ashini_image = face_recognition.load_image_file("img/durgaashini.jpg")
ashini_face_encoding = face_recognition.face_encodings(ashini_image)[0]

# Load a sample picture and learn how to recognize it.
harini_image = face_recognition.load_image_file("img/harini.jpg")
harini_face_encoding = face_recognition.face_encodings(harini_image)[0]

# Load a second sample picture and learn how to recognize it.
iu_image = face_recognition.load_image_file("img/iu.jpg")
iu_face_encoding = face_recognition.face_encodings(iu_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    ashini_face_encoding,
    harini_face_encoding,
    iu_face_encoding
]
known_face_names = [
    "Durgaashini"
    "Harini",
    "IU"
]

# Define the font to be used for the text
font = cv2.FONT_HERSHEY_SIMPLEX

@functools.lru_cache(maxsize=None)
def recognise_faces():
    # Initialize some variables
    process_this_frame = True


    # Initialize frame count and start time
    frame_count = 0
    start_time = time.time()

    # Get a reference to webcam #0 (the default one)
    capture_vid = cv2.VideoCapture(0)

    # Set the frame width and height
    capture_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Grab a single frame of video
        ret, frame = capture_vid.read()
        frame_count += 1

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            def detectRecog():
                # Initialize some variables
                face_locations = []
                face_encodings = []
                face_names = []

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                    process_this_frame = not process_this_frame


                    # Display the results
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Start the face detection and recognition thread
        thread = threading.Thread(target=detectRecog)
        thread.start()

        # Display the frame rate and elapsed time
        frame_rate = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {frame_rate:.2f}", (10, 50), font, 1, (255, 255, 255), 2)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    capture_vid.release()
    cv2.destroyAllWindows()

# Define a route for the index page
@appRecognition.route('/face_video')
def video_stream():
    return Response(recognise_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@appRecognition.route('/')
def index():
    return render_template('index.html')

# Run the web application
def run():
    sockApp.run(appRecognition)

if __name__ == '__main__':
    sockApp.run(appRecognition)