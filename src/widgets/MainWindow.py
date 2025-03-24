from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from src.ui.zastava import Ui_Zastava as MainUI
from src.threads.FaceRecog import VideoThreadFaceRecognition
from src.widgets.AddUser import AddUserWidget
from src.objects.Sound import SoundAnalyse
from src.db.Connect import DB

from loguru import logger

# Основной графический интерфейс
class Zastava(QtWidgets.QMainWindow):
    def __init__(self):
        super(Zastava, self).__init__()
        self.ui = MainUI()
        self.ui.setupUi(self)
        self.ui.addUserButton.clicked.connect(self.open_addUser)
        self.ui.vehicleRecognition.clicked.connect(self.show_page_1)
        self.ui.faceRecognition.clicked.connect(self.show_page_2)

        self.video_thread_face_recognition = VideoThreadFaceRecognition()
        self.video_thread_face_recognition.image_path_changed.connect(
            self.handle_image_path_changed
        )

        self.ui.OnOffVideoFaceRecogButton.clicked.connect( # OnOffVideoFaceRecogButton
            self.toggle_video_stream_face_recognition
        )

        self.widget = None
        # объявление класса анализа звукового окружения
        self.soundAnalyse = SoundAnalyse()
        self.soundAnalyse.update_text_signal.connect(self.display_yamnet_result)

    def update_audio(self):
        self.soundAnalyse.process_audio()

    def handle_image_path_changed(self, image_path_acc):
        self.image_path = image_path_acc
        self.load_user_data()

    def load_user_data(self):

        user_data = DB.get_user_by_image_path(image_path=self.image_path)

        if user_data is not None:
            surname = user_data[1]
            name = user_data[2]
            patronymic = user_data[3]
            rank = user_data[4]
            photo_path = user_data[5]

            # Загрузка фотографии
            photo = QPixmap(photo_path)

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
            self.ui.historyPhotoFaceRecog1.setPixmap(QPixmap())

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
