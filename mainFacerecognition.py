#final code

import face_recognition
import numpy as np
import os
import glob
import csv
from PIL import Image, ImageDraw


flagO = [0]
info = []
infonameD = []
infoflag = []

def training_image(known_people_folder):
    known_face_names = []
    known_face_encodings = []
    for file in known_people_folder:
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)
        known_face_names.append(basename)
        known_face_encodings.append(encodings[0])
    return known_face_names, known_face_encodings

path = os.path.join("training_images/", '*g')
known_people_folder = glob.glob(path)
known_face_names, known_face_encodings = training_image(known_people_folder)

#print(known_face_names)
#print(known_face_encodings)


def test_image(test_people_folder):
    test_face_encodings = []
    for file in test_people_folder:
        img = face_recognition.load_image_file(file)
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        pil_image = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_image)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            flag = 1
            print("Face Identified Value: " + str(flag) + ' name ' + name)

            infonameD.append(name)
            infoflag.append(flag)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # Draw a label with a name below the face
            #text_width, text_height = draw.textsize(name)
            #draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            #draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        pil_image.show()


path = os.path.join("test_images/", '*g')
test_people_folder = glob.glob(path)
test_image(test_people_folder)


#print(infonameD)
#Duplication maintain

infoname = []
for i in infonameD:
    if i not in infoname:
        infoname.append(i)

#print(infoname)
#print(infoflag)

with open('student_attencance.csv', mode='w', newline='') as attendance:
    attendance_write = csv.writer(attendance)
    info = [i for i in known_face_names if i not in infoname]
    attendance_write.writerow(["Student Name", "Attendance"])
    attendance_write.writerows(zip(infoname, infoflag))
    attendance_write.writerows(zip(info, flagO))

