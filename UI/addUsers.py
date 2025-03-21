from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AddUsersWidget(object):
    def setupUi(self, AddUsersWidget):
        AddUsersWidget.setObjectName("AddUsersWidget")
        AddUsersWidget.resize(1280, 720)
        AddUsersWidget.setMinimumSize(QtCore.QSize(1280, 720))
        AddUsersWidget.setMaximumSize(QtCore.QSize(1920, 1080))
        AddUsersWidget.setStyleSheet("background-color: rgb(26, 26, 26);")
        self.horizontalLayout = QtWidgets.QHBoxLayout(AddUsersWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.webCamAddFotoForm = QtWidgets.QFrame(AddUsersWidget)
        self.webCamAddFotoForm.setMinimumSize(QtCore.QSize(640, 0))
        self.webCamAddFotoForm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.webCamAddFotoForm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.webCamAddFotoForm.setObjectName("webCamAddFotoForm")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.webCamAddFotoForm)
        self.verticalLayout.setObjectName("verticalLayout")
        self.videoFromWebCam = QtWidgets.QLabel(self.webCamAddFotoForm)
        self.videoFromWebCam.setToolTipDuration(-1)
        self.videoFromWebCam.setStyleSheet(
            "background-color: rgb(46, 46, 46);\n" "border-radius: 10px;"
        )
        self.videoFromWebCam.setText("")
        self.videoFromWebCam.setObjectName("videoFromWebCam")
        self.verticalLayout.addWidget(self.videoFromWebCam)
        self.SaveFotoFromWebCamButton = QtWidgets.QPushButton(self.webCamAddFotoForm)
        self.SaveFotoFromWebCamButton.setStyleSheet(
            "QPushButton{\n"
            "background-color: rgb(35, 35, 35);\n"
            "color: #ffffff;\n"
            "border-style: outset;\n"
            "border-radius: 7px;\n"
            "font: bold 14px;\n"
            "min-width: 10em;\n"
            "padding: 6px;\n"
            "}\n"
            "QPushButton:hover{\n"
            "background-color: rgb(46, 46, 46);\n"
            "}\n"
            "QPushButton:unactive{\n"
            "background-color: rgb(46, 46, 46);\n"
            "}"
        )
        self.SaveFotoFromWebCamButton.setObjectName("SaveFotoFromWebCamButton")
        self.verticalLayout.addWidget(self.SaveFotoFromWebCamButton)
        self.horizontalLayout.addWidget(self.webCamAddFotoForm)
        self.AddTextInfoForm = QtWidgets.QFrame(AddUsersWidget)
        self.AddTextInfoForm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.AddTextInfoForm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.AddTextInfoForm.setObjectName("AddTextInfoForm")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.AddTextInfoForm)
        self.verticalLayout_2.setContentsMargins(0, -1, -1, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.VvediteDannie = QtWidgets.QLabel(self.AddTextInfoForm)
        self.VvediteDannie.setMaximumSize(QtCore.QSize(16777215, 50))
        self.VvediteDannie.setStyleSheet(
            "color: rgb(255, 255, 255);\n"
            "background-color: rgb(46, 46, 46);\n"
            "border-radius: 10px;\n"
            "font: bold 18px;\n"
            "min-width: 10em;\n"
            "padding: 6px; "
        )
        self.VvediteDannie.setAlignment(QtCore.Qt.AlignCenter)
        self.VvediteDannie.setObjectName("VvediteDannie")
        self.verticalLayout_2.addWidget(self.VvediteDannie)
        self.AddTextInfoFormBox = QtWidgets.QFrame(self.AddTextInfoForm)
        self.AddTextInfoFormBox.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.AddTextInfoFormBox.setFrameShadow(QtWidgets.QFrame.Raised)
        self.AddTextInfoFormBox.setObjectName("AddTextInfoFormBox")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.AddTextInfoFormBox)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.AddInfo = QtWidgets.QFrame(self.AddTextInfoFormBox)
        self.AddInfo.setMinimumSize(QtCore.QSize(600, 0))
        self.AddInfo.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.AddInfo.setFrameShadow(QtWidgets.QFrame.Raised)
        self.AddInfo.setObjectName("AddInfo")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.AddInfo)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setSpacing(12)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.RankForm = QtWidgets.QFrame(self.AddInfo)
        self.RankForm.setMaximumSize(QtCore.QSize(16777215, 100))
        self.RankForm.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.RankForm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.RankForm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.RankForm.setObjectName("RankForm")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.RankForm)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.Rank = QtWidgets.QLabel(self.RankForm)
        self.Rank.setMinimumSize(QtCore.QSize(240, 35))
        self.Rank.setMaximumSize(QtCore.QSize(200, 35))
        self.Rank.setStyleSheet(
            "color: rgb(255, 255, 255);\n" "font: bold 20px;\n" "min-width: 10em;\n" ""
        )
        self.Rank.setObjectName("Rank")
        self.verticalLayout_3.addWidget(self.Rank)
        self.RankComboBoxAdd = QtWidgets.QComboBox(self.RankForm)
        self.RankComboBoxAdd.setMinimumSize(QtCore.QSize(170, 35))
        self.RankComboBoxAdd.setMaximumSize(QtCore.QSize(16777215, 35))
        self.RankComboBoxAdd.setStyleSheet(
            "color: rgb(255, 255, 255);\n"
            "background-color: rgb(46, 46, 46);\n"
            "border-radius: 10px;\n"
            "font: bold 16px;\n"
        )

        self.RankComboBoxAdd.setCurrentText("")
        self.RankComboBoxAdd.setObjectName("RankComboBoxAdd")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.RankComboBoxAdd.addItem("")
        self.verticalLayout_3.addWidget(self.RankComboBoxAdd)
        self.verticalLayout_8.addWidget(self.RankForm)
        self.SurnameForm = QtWidgets.QFrame(self.AddInfo)
        self.SurnameForm.setMaximumSize(QtCore.QSize(16777215, 100))
        self.SurnameForm.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.SurnameForm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.SurnameForm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.SurnameForm.setObjectName("SurnameForm")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.SurnameForm)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.Surname = QtWidgets.QLabel(self.SurnameForm)
        self.Surname.setMinimumSize(QtCore.QSize(240, 35))
        self.Surname.setMaximumSize(QtCore.QSize(200, 35))
        self.Surname.setStyleSheet(
            "color: rgb(255, 255, 255);\n" "font: bold 20px;\n" "min-width: 10em;\n" ""
        )
        self.Surname.setObjectName("Surname")
        self.verticalLayout_5.addWidget(self.Surname)
        self.SurnameAddText = QtWidgets.QLineEdit(self.SurnameForm)
        self.SurnameAddText.setMinimumSize(QtCore.QSize(170, 35))
        self.SurnameAddText.setMaximumSize(QtCore.QSize(16777215, 35))
        self.SurnameAddText.setStyleSheet(
            "color: rgb(255, 255, 255);\n"
            "background-color: rgb(46, 46, 46);\n"
            "border-radius: 10px;\n"
            "font: bold 16px;\n"
            "padding-left: 10px;\n"
        )
        self.SurnameAddText.setObjectName("SurnameAddText")
        self.verticalLayout_5.addWidget(self.SurnameAddText)
        self.verticalLayout_8.addWidget(self.SurnameForm)
        self.NameForm = QtWidgets.QFrame(self.AddInfo)
        self.NameForm.setMaximumSize(QtCore.QSize(16777215, 100))
        self.NameForm.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.NameForm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.NameForm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.NameForm.setObjectName("NameForm")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.NameForm)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.Name = QtWidgets.QLabel(self.NameForm)
        self.Name.setMinimumSize(QtCore.QSize(240, 35))
        self.Name.setMaximumSize(QtCore.QSize(200, 35))
        self.Name.setStyleSheet(
            "color: rgb(255, 255, 255);\n" "font: bold 20px;\n" "min-width: 10em;\n" ""
        )
        self.Name.setObjectName("Name")
        self.verticalLayout_6.addWidget(self.Name)
        self.NameAddText = QtWidgets.QLineEdit(self.NameForm)
        self.NameAddText.setMinimumSize(QtCore.QSize(170, 35))
        self.NameAddText.setMaximumSize(QtCore.QSize(16777215, 35))
        self.NameAddText.setStyleSheet(
            "color: rgb(255, 255, 255);\n"
            "background-color: rgb(46, 46, 46);\n"
            "border-radius: 10px;\n"
            "font: bold 16px;\n"
            "padding-left: 10px;\n"
        )
        self.NameAddText.setObjectName("NameAddText")
        self.verticalLayout_6.addWidget(self.NameAddText)
        self.verticalLayout_8.addWidget(self.NameForm)
        self.FathersNameForm = QtWidgets.QFrame(self.AddInfo)
        self.FathersNameForm.setMaximumSize(QtCore.QSize(16777215, 100))
        self.FathersNameForm.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.FathersNameForm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.FathersNameForm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.FathersNameForm.setObjectName("FathersNameForm")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.FathersNameForm)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.FathersName = QtWidgets.QLabel(self.FathersNameForm)
        self.FathersName.setMinimumSize(QtCore.QSize(240, 35))
        self.FathersName.setMaximumSize(QtCore.QSize(200, 35))
        self.FathersName.setStyleSheet(
            "color: rgb(255, 255, 255);\n" "font: bold 20px;\n" "min-width: 10em;\n" ""
        )
        self.FathersName.setObjectName("FathersName")
        self.verticalLayout_7.addWidget(self.FathersName)
        self.FathersNameAddText = QtWidgets.QLineEdit(self.FathersNameForm)
        self.FathersNameAddText.setMinimumSize(QtCore.QSize(170, 35))
        self.FathersNameAddText.setMaximumSize(QtCore.QSize(16777215, 35))
        self.FathersNameAddText.setStyleSheet(
            "color: rgb(255, 255, 255);\n"
            "background-color: rgb(46, 46, 46);\n"
            "border-radius: 10px;\n"
            "font: bold 16px;\n"
            "padding-left: 10px;\n"
        )
        self.FathersNameAddText.setObjectName("FathersNameAddText")
        self.verticalLayout_7.addWidget(self.FathersNameAddText)
        self.verticalLayout_8.addWidget(self.FathersNameForm)
        self.verticalLayout_9.addWidget(self.AddInfo)
        spacerItem = QtWidgets.QSpacerItem(
            20, 231, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout_9.addItem(spacerItem)
        self.verticalLayout_2.addWidget(self.AddTextInfoFormBox)
        self.SaveFormButton = QtWidgets.QPushButton(self.AddTextInfoForm)
        self.SaveFormButton.setStyleSheet(
            "QPushButton{\n"
            "background-color: rgb(35, 35, 35);\n"
            "color: #ffffff;\n"
            "border-style: outset;\n"
            "border-radius: 7px;\n"
            "font: bold 14px;\n"
            "min-width: 10em;\n"
            "padding: 6px;\n"
            "}\n"
            "QPushButton:hover{\n"
            "background-color: rgb(46, 46, 46);\n"
            "}\n"
            "QPushButton:unactive{\n"
            "background-color: rgb(46, 46, 46);\n"
            "}"
        )
        self.SaveFormButton.setObjectName("SaveFormButton")
        self.verticalLayout_2.addWidget(self.SaveFormButton)
        self.horizontalLayout.addWidget(self.AddTextInfoForm)

        self.retranslateUi(AddUsersWidget)
        self.RankComboBoxAdd.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(AddUsersWidget)

    def retranslateUi(self, AddUsersWidget):
        _translate = QtCore.QCoreApplication.translate
        AddUsersWidget.setWindowTitle(_translate("AddUsersWidget", "Form"))
        self.SaveFotoFromWebCamButton.setText(
            _translate("AddUsersWidget", "Включить/выключить видеопоток ")
        )
        self.VvediteDannie.setText(_translate("AddUsersWidget", "Введите данные"))
        self.Rank.setText(_translate("AddUsersWidget", "Воинское звание"))
        self.RankComboBoxAdd.setItemText(0, _translate("AddUsersWidget", "Полковник"))
        self.RankComboBoxAdd.setItemText(
            1, _translate("AddUsersWidget", "Подполковник")
        )
        self.RankComboBoxAdd.setItemText(2, _translate("AddUsersWidget", "Майор"))
        self.RankComboBoxAdd.setItemText(3, _translate("AddUsersWidget", "Капитан"))
        self.RankComboBoxAdd.setItemText(
            4, _translate("AddUsersWidget", "Старший_лейтенант")
        )
        self.RankComboBoxAdd.setItemText(5, _translate("AddUsersWidget", "Лейтенант"))
        self.RankComboBoxAdd.setItemText(6, _translate("AddUsersWidget", "Старшина"))
        self.RankComboBoxAdd.setItemText(
            7, _translate("AddUsersWidget", "Старший_сержант")
        )
        self.RankComboBoxAdd.setItemText(8, _translate("AddUsersWidget", "Сержант"))
        self.RankComboBoxAdd.setItemText(
            9, _translate("AddUsersWidget", "Младший_сержант")
        )
        self.RankComboBoxAdd.setItemText(10, _translate("AddUsersWidget", "Ефрейтор"))
        self.RankComboBoxAdd.setItemText(11, _translate("AddUsersWidget", "Рядовой"))
        self.RankComboBoxAdd.setItemText(
            12, _translate("AddUsersWidget", "Гражданский_персонал")
        )
        self.Surname.setText(_translate("AddUsersWidget", "Фамилия"))
        self.Name.setText(_translate("AddUsersWidget", "Имя"))
        self.FathersName.setText(_translate("AddUsersWidget", "Отчество"))
        self.SaveFormButton.setText(_translate("AddUsersWidget", "Сохранить работника"))