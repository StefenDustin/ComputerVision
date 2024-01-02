import cv2
import os
import random
import numpy as np

average_accuracies = []

def train_test_model():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    PATH = './Dataset'
    person_name = os.listdir(PATH)

    train_img = []
    train_label = []
    test_img = []
    test_label = []

    for idx, person in enumerate(person_name):
        folder_path = PATH + "/" + person

        img_list = []
        label_list = []

        for img_name in os.listdir(folder_path):
            img_path = folder_path + "/" + img_name
            img_data = cv2.imread(img_path)

            img_list.append(img_data)
            label_list.append(idx)

        random.shuffle(img_list)

        TEST_SIZE = int(0.2*len(img_list))

        train_img.extend(img_list[TEST_SIZE:])
        test_img.extend(img_list[:TEST_SIZE])

        train_label.extend(label_list[TEST_SIZE:])
        test_label.extend(label_list[:TEST_SIZE])

    train_face_img = []
    train_face_label = []

    for image, label in zip(train_img, train_label):
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detect_faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.2, minNeighbors=9)

        if len(detect_faces) < 1:
            continue

        for face in detect_faces:
            x, y, h, w = face
            face_img = grayscale_img[y:y+h, x:x+w]
            train_face_img.append(face_img)
            train_face_label.append(label)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_img, np.array(train_face_label))

    face_recognizer.save('face_recognizer.xml')

    correct_predictions = 0
    total_predictions = 0

    for image, label in zip(test_img, test_label):
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detect_faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.2, minNeighbors=9)

        if len(detect_faces) < 1:
            continue

        for face in detect_faces:
            x, y, h, w = face
            face_img = grayscale_img[y:y+h, x:x+w]

            result, confidence = face_recognizer.predict(face_img)
            total_predictions += 1
            if result == label:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions else 0
    return accuracy, person_name


def predict(person_name):
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('face_recognizer.xml')
        model_file = 'face_recognizer.xml'
    except cv2.error:
        print("Error: Model not found. Please train the model first.")
        return

    input_image_path = input("Enter the absolute path of the image to be predicted: ")

    input_image = cv2.imread(input_image_path)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    grayscale_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    detect_faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.2, minNeighbors=9)

    for face in detect_faces:
        x, y, w, h = face
        face_img = grayscale_img[y:y+h, x:x+w]

        result, confidence = face_recognizer.predict(face_img)
        predicted_name = person_name[result]
        accuracy = (1 - confidence / 400) * 100  

        cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 10)
        text = f'Predicted: {predicted_name}, Confidence: {accuracy:.2f}%'
        cv2.putText(input_image, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

    cv2.imshow("Prediction Result", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

while True:
    print("Choose your option >>")
    print("1. Train and test model")
    print("2. Predict")
    print("3. Exit")

    choice = input("Choose your option >> ")

    if choice == '1':
        print("Training model...")
        accuracy, person_name = train_test_model()
        average_accuracies.append(accuracy)
        print("Training finished")
        print("Accuracy: ", accuracy * 100, "%")
        input("Press Enter to continue...")

    elif choice == '2':
        predict(person_name)
        input("Press Enter to continue...")

    elif choice == '3':
        print("Exiting program.")
        break  

    else:
        print("Invalid option. Please choose a valid option.")
        input("Press Enter to continue...")