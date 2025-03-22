import sys
import cv2
import os

import math
import threading
import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import keras
import face_recognition
import time
from pathlib import Path
import psycopg2
from psycopg2 import Error

import Yamnet.yamnet.params as params
import Yamnet.yamnet.yamnet as yamnet_model
from loguru import logger

from UI.addUsers import Ui_AddUsersWidget as AddUsersWidget
from UI.zastava import Ui_Zastava as ZastavaMainWindow

from PyQt5 import QtGui

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt


# класс для распознавания лиц с потока
class VideoThreadFaceRecognition(QThread):
    image_data_face_recognition = pyqtSignal(object)
    image_path_changed = pyqtSignal(str)

    def __init__(self, zastava_instance):
        super(VideoThreadFaceRecognition, self).__init__()
        self.zastava = zastava_instance
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
        people_folders = os.listdir("people")

        for folder in people_folders:
            folder_path = os.path.join("people", folder)
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
                    logger.error("ДОСТУП РАЗРЕШЕН") # Отправляем сигнал например к воротам

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
                        path_acc = Path.cwd().joinpath("people").joinpath(name)
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


# класс видеопотока в окне добавления лиц
class VideoThreadAddFace(QThread):
    image_data_face = pyqtSignal(object)

    video_capture_face_add = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.video_capture_face_add.read()

            if ret:
                rgb_image_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_image_face = cv2.resize(rgb_image_face, (1080, 1080))
                self.image_data_face.emit(resized_image_face)

        self.video_capture_face_add.release()

    def add_face_stop(self):
        self.terminate()


# Добавление юзера в БД
class AddUserWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.ui = AddUsersWidget()
        self.ui.setupUi(self)

        self.video_thread_face = VideoThreadAddFace()
        self.video_thread_face.image_data_face.connect(self.update_image_face)

        self.ui.SaveFotoFromWebCamButton.clicked.connect(self.toggle_add_face_potok)
        self.ui.SaveFormButton.clicked.connect(self.save_people)

    def update_image_face(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.videoFromWebCam.setPixmap(
            pixmap.scaled(
                self.ui.videoFromWebCam.size(), aspectRatioMode=Qt.KeepAspectRatio
            )
        )

    def start_add_face_potok(self):
        self.video_thread_face.start()
        logger.debug("start_add_face_potok")

    def stop_add_face_potok(self):
        self.video_thread_face.add_face_stop()
        logger.debug("stop_add_face_potok")

    def toggle_add_face_potok(self):
        if self.video_thread_face and self.video_thread_face.isRunning():
            logger.debug("toggle_add_face_potok - Выключаю поток добавления лиц")
            self.stop_add_face_potok()
        else:
            logger.debug("toggle_add_face_potok - Включаю поток добавления лиц")
            self.start_add_face_potok()

    def save_people(self):
        surname = self.ui.SurnameAddText.text()
        name = self.ui.NameAddText.text()
        patronymic = self.ui.FathersNameAddText.text()
        rank = self.ui.RankComboBoxAdd.currentText()

        # Создаем путь к папке на основе данных ФИО
        folder_path = os.path.join("people", f"{surname}_{name}_{patronymic}_{rank}")
        os.makedirs(folder_path, exist_ok=True)

        # Получить текущий кадр с видеопотока
        current_frame = self.ui.videoFromWebCam.pixmap().toImage()

        # Создать имя файла на основе временной метки
        filename = f"frame_{int(time.time())}.jpg"

        # Полный путь к файлу
        file_path = os.path.join(folder_path, filename)

        # Сохранить кадр в файл
        current_frame.save(file_path)

        # Вывести сообщение об успешном сохранении
        print(f"Кадр сохранен в {file_path}")

        conn = psycopg2.connect(
            host="localhost",
            database="zastava",
            user="postgres",
            password="postgres",
        )

        # Создание курсора
        cursor = conn.cursor()

        # Создание таблицы, если она не существует
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id SERIAL PRIMARY KEY,
                surname VARCHAR(255),
                name VARCHAR(255),
                patronymic VARCHAR(255),
                rank VARCHAR(255),
                photo_path VARCHAR(255)
            )
        """
        )

        # Вставка данных в таблицу
        cursor.execute(
            """
            INSERT INTO people (surname, name, patronymic, rank, photo_path)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (surname, name, patronymic, rank, file_path),
        )

        # Выполняем запрос на выборку всех строк из таблицы people
        cursor.execute("SELECT  *  FROM people")
        rows = cursor.fetchall()
        logger.debug(rows)

        # Подтверждение изменений
        conn.commit()

        # Закрытие соединения
        cursor.close()
        conn.close()

        self.ui.SurnameAddText.clear()
        self.ui.NameAddText.clear()
        self.ui.FathersNameAddText.clear()
        self.ui.RankComboBoxAdd.clearEditText()


# Основной графический интерфейс
class Zastava(QtWidgets.QMainWindow):
    def __init__(self):
        super(Zastava, self).__init__()
        self.ui = ZastavaMainWindow()
        self.ui.setupUi(self)
        self.ui.addUserButton.clicked.connect(self.open_addUser)
        self.ui.vehicleRecognition.clicked.connect(self.show_page_1)
        self.ui.faceRecognition.clicked.connect(self.show_page_2)

        self.video_thread_face_recognition = VideoThreadFaceRecognition(self)
        self.video_thread_face_recognition.image_path_changed.connect(
            self.handle_image_path_changed
        )

        self.ui.OnOffVideoFaceRecogButton.clicked.connect( # OnOffVideoFaceRecogButton
            self.toggle_video_stream_face_recognition
        )

        self.widget = None
        # объявление класса анализа звукового окружения
        self.soundAnalyse = soundAnalyse()
        self.soundAnalyse.update_text_signal.connect(self.display_yamnet_result)

    def update_audio(self):
        self.soundAnalyse.process_audio()

    def handle_image_path_changed(self, image_path_acc):
        self.image_path = image_path_acc
        self.load_user_data()

    def load_user_data(self):
        # Подключение к базе данных
        conn = psycopg2.connect(
            host="localhost",
            database="zastava",
            user="postgres",
            password="postgres",
        )
        # Создание курсора для выполнения SQL-запросов
        cursor = conn.cursor()

        # Выполнение SQL-запроса для получения информации о пользователе по пути
        cursor.execute(f"SELECT * FROM people WHERE photo_path='{self.image_path}'")
        user_data = cursor.fetchone()

        # Закрытие курсора и соединения с базой данных
        cursor.close()
        conn.close()

        if user_data is not None:
            surname = user_data[1]
            name = user_data[2]
            patronymic = user_data[3]
            rank = user_data[4]
            photo_path = user_data[5]

            # Загрузка фотографии
            photo = QtGui.QPixmap(photo_path)

            # Вывод данных о пользователе и фотографии в соответствующие виджеты
            self.ui.historyInfoFaceRecog1.setText(
                f"ФИО: {surname} {name} {patronymic}\nЗвание: {rank}"
            )
            self.ui.historyPhotoFaceRecog1.setPixmap(photo)

            # Обновление последнего распознанного человека в формах label
            self.last_recognized_person = {
                "surname": surname,
                "name": name,
                "patronymic": patronymic,
                "rank": rank,
                "photo": photo,
            }
        else:
            self.ui.historyInfoFaceRecog1.setText("Неизвестное лицо")
            self.ui.historyPhotoFaceRecog1.setPixmap(QtGui.QPixmap())

    def show_page_1(self):
        self.ui.stackedWidgetPage.setCurrentIndex(0)

    def show_page_2(self):
        self.ui.stackedWidgetPage.setCurrentIndex(1)

    def start_face_recognition_thread(self):
        self.video_thread_face_recognition.moveToThread(
            self.video_thread_face_recognition
        )
        self.video_thread_face_recognition.started.connect(
            self.video_thread_face_recognition.face_recognition_run
        )
        self.video_thread_face_recognition.image_data_face_recognition.connect(
            self.update_image_face_recognition
        )
        logger.debug("Видеопоток распознавания лиц включен")
        logger.debug(self.video_thread_face_recognition.start())

    def toggle_video_stream_face_recognition(self):
        if (
            self.video_thread_face_recognition
            and self.video_thread_face_recognition.isRunning()
        ):
            logger.debug("Выключаю распознавание лиц")
            self.stop_face_recognition_thread()
        else:
            self.start_face_recognition_thread()

    def stop_face_recognition_thread(self):
        self.video_thread_face_recognition.face_recognition_stop()

    # Открытие виджета добавления пользователя
    def open_addUser(self):
        if self.widget is None:
            self.widget = AddUserWidget()
        self.widget.show()

    # Отображение результата работы yamnet
    def display_yamnet_result(self, result):
        self.ui.YamnetTextBox.setText(result)

    # Отображение результатов face_recog
    def update_image_face_recognition(self, image):
        h, w, ch = image.shape
        center_x = w // 2
        center_y = h // 2
        box_size = 200
        x1 = center_x - box_size // 2
        y1 = center_y - box_size // 2
        x2 = center_x + box_size // 2
        y2 = center_y + box_size // 2
        cropped_image = image[y1:y2, x1:x2]
        # остальной код для отображения обработанной области
        bytes_per_line = ch * box_size
        q_image = QImage(
            bytes(cropped_image.data),
            box_size,
            box_size,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(q_image)
        self.ui.videoPotokFaceRecog.setPixmap(
            pixmap.scaled(
                self.ui.videoPotokFaceRecog.size(), aspectRatioMode=Qt.KeepAspectRatio
            )
        )
        self.ui.videoPotokFaceRecog.setScaledContents(True)

class soundAnalyse(QtCore.QObject):
    update_text_signal = QtCore.pyqtSignal(str)

    def process_audio(self):
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights("Yamnet/yamnet/yamnet.h5")
        yamnet_classes = yamnet_model.class_names("Yamnet/yamnet/yamnet_class_map.csv")

        frame_len = int(params.SAMPLE_RATE * 1)  # 1sec

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=params.SAMPLE_RATE,
            input=True,
            frames_per_buffer=frame_len,
        )

        while True:
            # data read
            data = stream.read(frame_len, exception_on_overflow=False)

            # byte --> float
            frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

            # model prediction
            scores, melspec = yamnet.predict(np.reshape(frame_data, [1, -1]), steps=1)
            prediction = np.mean(scores, axis=0)

            top5_i = np.argsort(prediction)[::-1][:2]

            # append result to list
            result = "Текущее событие:\n" + "".join(
                " {:12s}: {:.3f}".format(yamnet_classes[i], prediction[i])
                for i in top5_i
            )
            self.update_text_signal.emit(result)
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    application = Zastava()

    # создание потока звукоанализа
    yamnet_thread = threading.Thread(target=application.update_audio)
    yamnet_thread.start()

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

    application.show()

    sys.exit(app.exec_())
