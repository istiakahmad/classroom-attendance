import cv2
import numpy as np
import os
from PIL import Image
import glob

def training():
    print("Begin")
    recognizer = cv2.face.EigenFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

    def preparing_training_data():

        # Path for face image database
        path = 'training_data'
        dirs = os.listdir(path)

        # list to hold all subject faces
        faces = []
        # list to hold labels for all subjects
        labels = []

        folder_count = 0
        width_d, height_d = 280, 280  # Declare your own width and height

        for dir_name in dirs:

            folder_count = folder_count + 1
            label = folder_count

            subject_dir_path = path + "/" + dir_name

            subject_images_names = os.listdir(subject_dir_path)

            imagePaths = [os.path.join(subject_dir_path, f) for f in os.listdir(subject_dir_path)]

            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                face = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in face:
                    faces.append(cv2.resize(img_numpy[y:y + h, x:x + w], (width_d, height_d)))
                    labels.append(label)
                #print("3")
            #print("2")
        #print("1")
        return faces, labels
    #print("0")
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, labels = preparing_training_data()
    recognizer.train(faces, np.array(labels))

    # Save the model into trainer/trainer.yml
    recognizer.write('EigenTrainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(labels))))


def recognization(test_people_folder):
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.read('EigenTrainer.yml')
    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    label = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'biden', 'obama']

    # Initialize and start realtime video capture
    # cam = cv2.VideoCapture("video\gates.mp4")
    width_d, height_d = 280, 280  # Declare your own width and height

    for file in test_people_folder:
        img = cv2.imread(file)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label, confidence = recognizer.predict(cv2.resize(gray[y:y + h, x:x + w], (width_d, height_d)))

            #print(label)

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 55):
                label = names[label]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                label = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            print(label)
            cv2.putText(img, str(label), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

    # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

training()
path = os.path.join("test_images/", '*g')
test_people_folder = glob.glob(path)
recognization(test_people_folder)

