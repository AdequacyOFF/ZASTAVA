import sys

import threading
from PyQt5 import QtWidgets
from src.widgets.MainWindow import Zastava
from src.db.Connect import DB
from PyQt5.QtCore import qInstallMessageHandler, QtCriticalMsg, QtFatalMsg

from loguru import logger


def qt_message_handler(mode, context, message):
    if mode == QtCriticalMsg or mode == QtFatalMsg:
        logger.error(f"Qt Ошибка: {message} (файл: {context.file}, строка: {context.line})")


if __name__ == "__main__":
    qInstallMessageHandler(qt_message_handler)
    app = QtWidgets.QApplication(sys.argv)

    db = DB()
    application = Zastava(db)

    # создание потока звукоанализа
    yamnet_thread = threading.Thread(target=application.update_audio)
    yamnet_thread.start()

    application.show()

    sys.exit(app.exec_())


