import sys
import cv2
import os
import math
import threading
import pyaudio
import librosa
import numpy as np
import face_recognition
import time
from pathlib import Path
import psycopg2
from psycopg2 import Error
from loguru import logger

# Импорты для YAMNet
import Yamnet.yamnet.params as yamnet_params
import Yamnet.yamnet.yamnet as yamnet_model

from UI.addUsers import Ui_AddUsersWidget as AddUsersWidget
from UI.zastava import Ui_Zastava as ZastavaMainWindow

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class VideoThreadFaceRecognition(QThread):
    image_data_face_recognition = pyqtSignal(object)
    image_path_changed = pyqtSignal(str)

    def __init__(self, zastava_instance):
        super().__init__()
        self.zastava = zastava_instance
        self.ThreadActive = False
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_this_frame = True

    def load_face_database(self):
        """Загружает и валидирует базу лиц из папки 'people'"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists("people"):
            os.makedirs("people")
            logger.warning("Папка 'people' не найдена, создана новая")
            return

        people_folders = os.listdir("people")
        if not people_folders:
            logger.warning("В папке 'people' нет данных о лицах")
            return

        invalid_images = []
        
        for folder in people_folders:
            folder_path = os.path.join("people", folder)
            if os.path.isdir(folder_path):
                for image in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image)
                    try:
                        face_image = face_recognition.load_image_file(image_path)
                        face_encoding = face_recognition.face_encodings(face_image)
                        
                        if face_encoding:
                            self.known_face_encodings.append(face_encoding[0])
                            self.known_face_names.append(folder)
                            logger.debug(f"Загружено лицо: {folder} ({image_path})")
                        else:
                            invalid_images.append(image_path)
                            logger.warning(f"Не найдено лиц на изображении: {image_path}")
                            
                    except Exception as e:
                        logger.error(f"Ошибка загрузки {image_path}: {str(e)}")
                        invalid_images.append(image_path)

        if invalid_images:
            logger.warning(f"Найдено {len(invalid_images)} изображений без распознаваемых лиц")

    def run(self):
        """Основной цикл распознавания лиц"""
        self.ThreadActive = True
        self.load_face_database()
        
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logger.error("Не удалось открыть камеру")
            return

        try:
            while self.ThreadActive:
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Обрабатываем каждый второй кадр для увеличения производительности
                if self.process_this_frame:
                    # Уменьшаем кадр для ускорения обработки
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB
                    
                    # Находим все лица в текущем кадре видео
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []
                    image_path_acc = ""
                    
                    for face_encoding in face_encodings:
                        # Проверяем, есть ли совпадение с известными лицами
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"
                        confidence = "N/A"

                        # Если найдены совпадения, используем самое близкое
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = self.known_face_names[first_match_index]
                            
                            # Вычисляем расстояние до всех известных лиц
                            face_distances = face_recognition.face_distance(
                                self.known_face_encodings, face_encoding
                            )
                            best_match_index = np.argmin(face_distances)
                            confidence = face_confidence(face_distances[best_match_index])
                            
                            # Получаем путь к фото
                            path_acc = Path("people") / name
                            if path_acc.exists():
                                images = list(path_acc.iterdir())
                                if images:
                                    image_path_acc = str(images[0])

                        face_names.append(f"{name} ({confidence})")

                    # Отображаем результаты
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Масштабируем координаты лица обратно к исходному размеру
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Рисуем прямоугольник вокруг лица
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                        # Рисуем подпись с именем
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

                    # Отправляем кадр в интерфейс
                    self.image_data_face_recognition.emit(frame)
                    if image_path_acc:
                        self.image_path_changed.emit(image_path_acc)
                
                self.process_this_frame = not self.process_this_frame
                
        finally:
            video_capture.release()
            logger.debug("Поток распознавания лиц остановлен")

    def stop(self):
        """Останавливает поток"""
        self.ThreadActive = False
        self.quit()
        self.wait()

class VideoThreadAddFace(QThread):
    image_data_face = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self._running = False
        self.video_capture = None

    def run(self):
        """Захват видео с камеры для добавления нового лица"""
        self._running = True
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            logger.error("Не удалось открыть камеру")
            return

        try:
            while self._running:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(rgb_image, (1080, 1080))
                self.image_data_face.emit(resized_image)
        finally:
            if self.video_capture:
                self.video_capture.release()
            logger.debug("Поток добавления лица остановлен")

    def stop(self):
        """Останавливает поток"""
        self._running = False
        self.quit()
        self.wait()


class SoundAnalyzer(QThread):
    update_text_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._running = False
        self.yamnet = None
        self.yamnet_classes = None
        
    def initialize_yamnet(self):
        """Инициализация модели YAMNet"""
        try:
            self.yamnet = yamnet_model.yamnet_frames_model(yamnet_params)
            self.yamnet.load_weights("Yamnet/yamnet/yamnet.h5")
            self.yamnet_classes = yamnet_model.class_names("Yamnet/yamnet/yamnet_class_map.csv")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации YAMNet: {str(e)}")
            return False
        
    def run(self):
        """Анализ звукового окружения"""
        if not self.initialize_yamnet():
            self.update_text_signal.emit("Ошибка инициализации анализатора звука")
            return
            
        self._running = True
        frame_len = int(yamnet_params.SAMPLE_RATE * 1)  # 1 секунда аудио

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=yamnet_params.SAMPLE_RATE,
            input=True,
            frames_per_buffer=frame_len,
        )

        while self._running:
            try:
                # Чтение аудиоданных
                data = stream.read(frame_len, exception_on_overflow=False)

                # Конвертация в float
                frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

                # Предсказание модели
                scores, _ = self.yamnet.predict(np.reshape(frame_data, [1, -1]), steps=1)
                prediction = np.mean(scores, axis=0)
                top5_i = np.argsort(prediction)[::-1][:2]  # Топ-2 результата

                # Формирование строки результата
                result = "Текущее событие:\n" + "\n".join(
                    f"{self.yamnet_classes[i]}: {prediction[i]*100:.1f}%"
                    for i in top5_i
                )
                
                self.update_text_signal.emit(result)
                
            except Exception as e:
                logger.error(f"Ошибка анализа звука: {str(e)}")
                continue

        # Очистка ресурсов
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    def stop(self):
        """Останавливает анализ звука"""
        self._running = False
        self.quit()
        self.wait()


class AddUserWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = AddUsersWidget()
        self.ui.setupUi(self)
        self.video_thread = VideoThreadAddFace()
        self.video_thread.image_data_face.connect(self.update_image)
        
        self.ui.SaveFotoFromWebCamButton.clicked.connect(self.toggle_video_stream)
        self.ui.SaveFormButton.clicked.connect(self.save_user)

    def update_image(self, image):
        """Обновляет изображение в интерфейсе"""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.videoFromWebCam.setPixmap(
            pixmap.scaled(self.ui.videoFromWebCam.size(), Qt.KeepAspectRatio)
        )

    def toggle_video_stream(self):
        """Включает/выключает видеопоток"""
        if self.video_thread.isRunning():
            logger.debug("Останавливаю видеопоток")
            self.video_thread.stop()
            self.ui.SaveFotoFromWebCamButton.setText("Включить камеру")
        else:
            logger.debug("Запускаю видеопоток")
            self.video_thread.start()
            self.ui.SaveFotoFromWebCamButton.setText("Выключить камеру")

    def save_user(self):
        """Сохраняет нового пользователя в базу"""
        if self.video_thread.isRunning():
            self.video_thread.stop()

        if not self.ui.videoFromWebCam.pixmap():
            logger.error("Нет изображения для сохранения")
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет изображения для сохранения")
            return

        surname = self.ui.SurnameAddText.text().strip()
        name = self.ui.NameAddText.text().strip()
        patronymic = self.ui.FathersNameAddText.text().strip()
        rank = self.ui.RankComboBoxAdd.currentText().strip()

        if not all([surname, name, patronymic, rank]):
            logger.error("Не все поля заполнены")
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Заполните все поля")
            return

        folder_name = f"{surname}_{name}_{patronymic}_{rank}"
        folder_path = os.path.join("people", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        filename = f"face_{int(time.time())}.jpg"
        file_path = os.path.join(folder_path, filename)

        if not self.ui.videoFromWebCam.pixmap().save(file_path):
            logger.error(f"Не удалось сохранить изображение: {file_path}")
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Не удалось сохранить фото")
            return

        try:
            conn = psycopg2.connect(
                host="localhost",
                database="zastava",
                user="postgres",
                password="2473",
            )
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO people (surname, name, patronymic, rank, photo_path)
                VALUES (%s, %s, %s, %s, %s)
            """, (surname, name, patronymic, rank, file_path))

            conn.commit()
            logger.success(f"Пользователь сохранен: {surname} {name}")
            QtWidgets.QMessageBox.information(self, "Успех", "Пользователь сохранен!")

        except Error as e:
            logger.error(f"Ошибка базы данных: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка базы данных: {str(e)}")
        finally:
            if conn:
                cursor.close()
                conn.close()

        # Очищаем форму
        self.ui.SurnameAddText.clear()
        self.ui.NameAddText.clear()
        self.ui.FathersNameAddText.clear()
        self.ui.RankComboBoxAdd.setCurrentIndex(0)

    def closeEvent(self, event):
        """Останавливает поток при закрытии окна"""
        if self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()


class Zastava(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = ZastavaMainWindow()
        self.ui.setupUi(self)
        
        # Инициализация потока распознавания лиц
        self.face_recognition_thread = VideoThreadFaceRecognition(self)
        self.face_recognition_thread.image_data_face_recognition.connect(
            self.update_face_recognition_image)
        self.face_recognition_thread.image_path_changed.connect(
            self.handle_image_path_changed)
            
        # Инициализация анализатора звука
        self.sound_analyzer = SoundAnalyzer()
        self.sound_analyzer.update_text_signal.connect(self.update_sound_text)
        
        # Настройка кнопок
        self.ui.addUserButton.clicked.connect(self.open_add_user)
        self.ui.vehicleRecognition.clicked.connect(self.show_page_1)
        self.ui.faceRecognition.clicked.connect(self.show_page_2)
        self.ui.OnOffVideoFaceRecogButton.clicked.connect(self.toggle_face_recognition)
        self.ui.OnOffAllRecogButton.clicked.connect(self.toggle_sound_analysis)

        self.add_user_widget = None
        
        # Запуск анализа звука при старте
        self.sound_analyzer.start()
        self.ui.OnOffAllRecogButton.setText("Выключить анализ звука")

    def toggle_sound_analysis(self):
        """Включает/выключает анализ звука"""
        if self.sound_analyzer.isRunning():
            logger.debug("Останавливаю анализ звука")
            self.sound_analyzer.stop()
            self.ui.OnOffAllRecogButton.setText("Включить анализ звука")
            self.ui.YamnetTextBox.setText("Анализ звука выключен")
        else:
            logger.debug("Запускаю анализ звука")
            self.sound_analyzer.start()
            self.ui.OnOffAllRecogButton.setText("Выключить анализ звука")

    def update_sound_text(self, text):
        """Обновляет текст с результатами анализа звука"""
        self.ui.YamnetTextBox.setText(text)

    def toggle_face_recognition(self):
        """Включает/выключает распознавание лиц"""
        if self.face_recognition_thread.isRunning():
            logger.debug("Останавливаю распознавание лиц")
            self.face_recognition_thread.stop()
            self.ui.OnOffVideoFaceRecogButton.setText("Включить распознавание")
            self.ui.videoPotokFaceRecog.clear()
        else:
            logger.debug("Запускаю распознавание лиц")
            self.face_recognition_thread.start()
            self.ui.OnOffVideoFaceRecogButton.setText("Выключить распознавание")

    def update_face_recognition_image(self, image):
        """Обновляет изображение с распознанными лицами"""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.videoPotokFaceRecog.setPixmap(
            pixmap.scaled(self.ui.videoPotokFaceRecog.size(), Qt.KeepAspectRatio))
        self.ui.videoPotokFaceRecog.setScaledContents(True)

    def handle_image_path_changed(self, image_path):
        """Обрабатывает изменение пути к изображению"""
        self.image_path = image_path
        self.load_user_data()

    def load_user_data(self):
        """Загружает данные пользователя из базы"""
        try:
            conn = psycopg2.connect(
                host="localhost",
                database="zastava",
                user="postgres",
                password="2473",
            )
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM people WHERE photo_path = %s", (self.image_path,))
            user_data = cursor.fetchone()
            
            if user_data:
                surname, name, patronymic, rank, photo_path = user_data[1:6]
                self.ui.historyInfoFaceRecog1.setText(
                    f"ФИО: {surname} {name} {patronymic}\nЗвание: {rank}")
                
                if os.path.exists(photo_path):
                    self.ui.historyPhotoFaceRecog1.setPixmap(QPixmap(photo_path))
                else:
                    logger.warning(f"Изображение не найдено: {photo_path}")
                    self.ui.historyPhotoFaceRecog1.clear()
            else:
                self.ui.historyInfoFaceRecog1.setText("Неизвестное лицо")
                self.ui.historyPhotoFaceRecog1.clear()
                
        except Error as e:
            logger.error(f"Ошибка базы данных: {str(e)}")
        finally:
            if conn:
                cursor.close()
                conn.close()

    def show_page_1(self):
        """Переключает на страницу обнаружения объектов"""
        self.ui.stackedWidgetPage.setCurrentIndex(0)
        self.ui.vehicleRecognition.setStyleSheet(
            "QPushButton{background-color: rgb(46, 46, 46); color: #ffffff; border-radius: 7px; font: bold 14px;}"
        )
        self.ui.faceRecognition.setStyleSheet(
            "QPushButton{background-color: rgb(35, 35, 35); color: #ffffff; border-radius: 7px; font: bold 14px;}"
        )

    def show_page_2(self):
        """Переключает на страницу контроля доступа"""
        self.ui.stackedWidgetPage.setCurrentIndex(1)
        self.ui.vehicleRecognition.setStyleSheet(
            "QPushButton{background-color: rgb(35, 35, 35); color: #ffffff; border-radius: 7px; font: bold 14px;}"
        )
        self.ui.faceRecognition.setStyleSheet(
            "QPushButton{background-color: rgb(46, 46, 46); color: #ffffff; border-radius: 7px; font: bold 14px;}"
        )

    def open_add_user(self):
        """Открывает окно добавления пользователя"""
        if self.add_user_widget is None:
            self.add_user_widget = AddUserWidget()
        self.add_user_widget.show()

    def closeEvent(self, event):
        """Останавливает все потоки при закрытии приложения"""
        if self.face_recognition_thread.isRunning():
            self.face_recognition_thread.stop()
        if self.sound_analyzer.isRunning():
            self.sound_analyzer.stop()
        event.accept()


def face_confidence(face_distance, face_match_threshold=0.6):
    """Рассчитывает уверенность распознавания лица в процентах"""
    range_val = 1.0 - face_match_threshold
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + "%"


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Создаем таблицу, если ее нет
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="zastava",
            user="postgres",
            password="2473",
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id SERIAL PRIMARY KEY,
                surname VARCHAR(255),
                name VARCHAR(255),
                patronymic VARCHAR(255),
                rank VARCHAR(255),
                photo_path VARCHAR(255)
            )
        """)
        conn.commit()
    except Error as e:
        logger.error(f"Ошибка инициализации базы данных: {str(e)}")
    finally:
        if conn:
            cursor.close()
            conn.close()

    window = Zastava()
    window.show()
    sys.exit(app.exec_())