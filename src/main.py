import sys
import threading
from PyQt5 import QtWidgets
from widgets.MainWindow import Zastava

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    application = Zastava()

    # создание потока звукоанализа
    yamnet_thread = threading.Thread(target=application.update_audio)
    yamnet_thread.start()

    application.show()

    sys.exit(app.exec_())
