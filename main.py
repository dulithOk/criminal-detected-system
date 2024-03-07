import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# video_capture = cv2.VideoCapture(0)

# Load known faces and encodings
jobs_image = face_recognition.load_image_file("image/jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

tesla_image = face_recognition.load_image_file("image/tesla.jpg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

max_image = face_recognition.load_image_file("image/max.jpg")
max_encoding = face_recognition.face_encodings(max_image)[0]

enri_image = face_recognition.load_image_file("image/enri.jpg")
enri_encoding = face_recognition.face_encodings(enri_image)[0]

tay_image = face_recognition.load_image_file("image/tay.jpg")
tay_encoding = face_recognition.face_encodings(tay_image)[0]

known_face_encodings = [jobs_encoding, tesla_encoding,max_encoding,enri_encoding,tay_encoding]
known_faces_names = ["jobs", "tesla","max","enri","tay"]

students = known_faces_names.copy()

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize CSV writer
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file_name = current_date + '.csv'
csv_file = open(csv_file_name, 'w+', newline='')
csv_writer = csv.writer(csv_file)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Check if the frame is valid
    if not ret or frame is None:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Perform face recognition
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = ""

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Check if it's a match
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        # Update attendance and display on frame
        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (150, 0, 0)
            thickness = 3
            lineType = 2

            cv2.putText(frame, f"{name} detected", bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                        lineType)

            if name in students:
                students.remove(name)
                print(students)
                current_time = now.strftime("%H-%M-%S")
                csv_writer.writerow([name, current_time])

    # Display the resulting frame
    cv2.imshow("criminal system", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close CSV file
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()



