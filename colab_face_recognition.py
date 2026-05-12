import os

import cv2
import numpy as np
from google.colab import drive
from google.colab.patches import cv2_imshow


# Google Drive bağlantısı
DRIVE_MOUNT_POINT = "/content/drive"
TRAIN_DIR = "/content/drive/MyDrive/FaceProject/oyuncu_yuzleri"
TEST_DIR = "/content/drive/MyDrive/FaceProject/test_images"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def get_lbph_recognizer():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face bulunamadı. Colab'da 'pip install opencv-contrib-python' çalıştırın."
        )
    return cv2.face.LBPHFaceRecognizer_create()


def detect_faces(gray_image, cascade):
    return cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )


def prepare_training_data(train_dir, cascade):
    face_samples = []
    labels = []
    label_to_name = {}
    name_to_label = {}

    for person_name in sorted(os.listdir(train_dir)):
        person_path = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in name_to_label:
            new_label = len(name_to_label)
            name_to_label[person_name] = new_label
            label_to_name[new_label] = person_name

        label = name_to_label[person_name]
        for file_name in sorted(os.listdir(person_path)):
            if not file_name.lower().endswith(".jpg"):
                continue

            image_path = os.path.join(person_path, file_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, cascade)
            for (x, y, w, h) in faces:
                face_roi = gray[y : y + h, x : x + w]
                face_samples.append(face_roi)
                labels.append(label)

    if not face_samples:
        raise RuntimeError("Eğitim için yüz bulunamadı. Klasör ve fotoğrafları kontrol edin.")

    return face_samples, np.array(labels), label_to_name


def annotate_test_images(test_dir, cascade, recognizer, label_to_name, threshold=70):
    for file_name in sorted(os.listdir(test_dir)):
        if not file_name.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(test_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, cascade)

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            predicted_label, confidence = recognizer.predict(face_roi)

            # LBPH'de daha düşük confidence daha iyi eşleşmeyi gösterir.
            if confidence <= threshold and predicted_label in label_to_name:
                text = label_to_name[predicted_label]
                color = (0, 255, 0)
            else:
                text = "Kayıtlı Değil"
                color = (0, 0, 255)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image,
                text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        print(f"Test görüntüsü: {file_name}")
        cv2_imshow(image)


def main():
    drive.mount(DRIVE_MOUNT_POINT)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError("HaarCascade modeli yüklenemedi.")

    recognizer = get_lbph_recognizer()
    face_samples, labels, label_to_name = prepare_training_data(TRAIN_DIR, cascade)
    recognizer.train(face_samples, labels)

    annotate_test_images(TEST_DIR, cascade, recognizer, label_to_name)


if __name__ == "__main__":
    main()
