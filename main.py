import face_recognition
import numpy as np
import os
import glob
import csv

def training_image(known_people_folder):
    known_names = []
    known_face_encodings = []
    for file in known_people_folder:
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)
        known_names.append(basename)
        known_face_encodings.append(encodings[0])
    return known_names, known_face_encodings

path = os.path.join("training_images/", '*g')
known_people_folder = glob.glob(path)
known_face_names, known_face_encodings = training_image(known_people_folder)


def test_image(test_people_folder):
    test_face_encodings = []
    face_locations = []
    for file in test_people_folder:
        img = face_recognition.load_image_file(file)
        face_locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, face_locations)
        test_face_encodings.append(encodings[0])
    return test_face_encodings

path = os.path.join("test_images/", '*g')
test_people_folder = glob.glob(path)
test_face_encodings = test_image(test_people_folder)

flagO = [0]
info = []
infoname = []
infoflag = []

for face_encoding in test_face_encodings:
        # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        flag = 1
        print("Face Identified Value: " + str(flag) + ' name ' + name)
        infoname.append(name)
        infoflag.append(flag)

with open('student_attencance.csv', mode='w', newline='') as attendance:
    attendance_write = csv.writer(attendance)
    info = [i for i in known_face_names if i not in infoname]
    attendance_write.writerow(["Student Name", "Attendance"])
    attendance_write.writerows(zip(infoname, infoflag))
    attendance_write.writerows(zip(info, flagO))
