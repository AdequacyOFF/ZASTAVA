from PyQt5.QtCore import QThread, pyqtSignal
import cv2


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
