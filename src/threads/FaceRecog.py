from PyQt5.QtCore import QThread, pyqtSignal
import os
import face_recognition
from loguru import logger
from pathlib import Path
import cv2
import numpy as np
import math


# класс для распознавания лиц с потока
class VideoThreadFaceRecognition(QThread):
    image_data_face_recognition = pyqtSignal(object)
    image_path_changed = pyqtSignal(str)

    def __init__(self):
        super(VideoThreadFaceRecognition, self).__init__()
        self.unknown_face_detected = False

    def face_recognition_run(self):
        self.ThreadActive = True
        # Загрузка базы данных лиц
        face_locations = []
        face_encodings = []
        face_names = []
        known_face_encodings = []
        known_face_names = []
        consecutive_detections = 0
        people_folders = os.listdir("../../people")

        for folder in people_folders:
            folder_path = os.path.join("../../people", folder)
            if os.path.isdir(folder_path):
                for image in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image)
                    face_image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(face_image)[0]

                    known_face_encodings.append(face_encoding)
                    known_face_names.append(folder)
                    logger.debug(folder)
                    logger.debug(image_path)

        video_capture_face_recognition = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = video_capture_face_recognition.read()

            if ret:
                small_frame = cv2.resize(frame, (330, 330), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )

                if len(face_encodings) > 0:
                    consecutive_detections += 1
                else:
                    consecutive_detections = 0
                if consecutive_detections >= 10:
                    logger.error("ДОСТУП РАЗРЕШЕН")  # Отправляем сигнал например к воротам

                face_names = []
                image_path_acc = ""  # Переменная для хранения пути к фотографии
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding
                    )
                    name = "unknown"
                    confidence = "unknown"

                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                        # Получите имя первого человека и уверенность
                        logger.debug(name)
                        logger.debug(confidence)

                        # Сохраните путь к фотографии
                        path_acc = Path.cwd().joinpath("../../people").joinpath(name)
                        for image in path_acc.iterdir():
                            image_path_acc = "people" + "\\" + name + "\\" + image.name

                    face_names.append(f"{name} ({confidence})")

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 1
                    right *= 1
                    bottom *= 1
                    left *= 1

                    cv2.rectangle(
                        rgb_small_frame, (left, top), (right, bottom), (0, 255, 0), 1
                    )
                self.image_data_face_recognition.emit(rgb_small_frame)
                self.image_path_changed.emit(
                    image_path_acc
                )  # Отправьте путь к фотографии

    def face_recognition_stop(self):
        self.ThreadActive = False
        self.terminate()

def face_confidence(face_distance, face_match_threshold=0.6):
    range = 1.0 - face_match_threshold
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (
            linear_val
            + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
        ) * 100
        return str(round(value, 2)) + "%"