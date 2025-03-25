from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from loguru import logger
import os
import time

from src.threads.AddFaceVideo import VideoThreadAddFace
from src.ui.addUsers import Ui_AddUsersWidget as AddUsersWidget
from src.utils.ThreadClose import end_thread


# Добавление юзера в БД
class AddUserWidget(QWidget):
    def __init__(self, database):
        super().__init__()
        self.database = database

        self.ui = AddUsersWidget()
        self.ui.setupUi(self)

        self.video_thread_face = VideoThreadAddFace()
        self.video_thread_face.image_data_face.connect(self.update_image_face)

        self.ui.SaveFotoFromWebCamButton.clicked.connect(self.toggle_add_face_thread)
        self.ui.SaveFormButton.clicked.connect(self.save_people)

    def closeEvent(self, event) -> None:
        # Завершение работы потока, если он активен
        end_thread(self.video_thread_face)
        self.close()
        logger.debug("Все процессы завершены.")
        event.accept()  # Подтверждаем закрытие

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

    def start_add_face_thread(self):
        self.video_thread_face.start()
        logger.debug("start_add_face_potok")

    def stop_add_face_thread(self):
        self.video_thread_face.add_face_stop()
        self.close()
        logger.debug("stop_add_face_potok")

    def toggle_add_face_thread(self):
        if self.video_thread_face and self.video_thread_face.isRunning():
            logger.debug("toggle_add_face_potok - Выключаю поток добавления лиц")
            self.stop_add_face_thread()
        else:
            logger.debug("toggle_add_face_potok - Включаю поток добавления лиц")
            self.start_add_face_thread()

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
        if current_frame.save(file_path):
            # Вывести сообщение об успешном сохранении
            print(f"Кадр сохранен в {file_path}")
        else:
            raise Exception("Ошибка при сохранении пользователя")

        self.database.add_user(surname, name, patronymic, rank, file_path)
        # Выполняем запрос на выборку всех строк из таблицы people
        rows = self.database.get_all_users()
        logger.debug(rows)

        self.ui.SurnameAddText.clear()
        self.ui.NameAddText.clear()
        self.ui.FathersNameAddText.clear()
        self.ui.RankComboBoxAdd.clearEditText()

        self.stop_add_face_thread()
