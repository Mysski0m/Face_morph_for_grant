from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QInputDialog
from data_from_json import PATH_TO_IMPORT_IMAGES, PATH_TO_SAVE_RESULT


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()

    def filename1_ButtonPressed(self):
        res = QFileDialog.getOpenFileName(
            self, "Open first image", PATH_TO_IMPORT_IMAGES, ""
        )
        img1_name = str(res[0])

        return img1_name

    def filename2_ButtonPressed_2(self, num):
        res = QFileDialog.getOpenFileName(
            self, f"Open second image ({num + 1})", PATH_TO_IMPORT_IMAGES, ""
        )
        img2_name = str(res[0])

        return img2_name

    def save_file(self, num):
        res = QFileDialog.getSaveFileName(
            self, f"Save result ({num + 1})", PATH_TO_SAVE_RESULT, "JPG File (*.jpg)"
        )
        return str(res[0])

    def input_alpha(self):
        res = QInputDialog.getText(
            self,
            "Параметр alpha",
            "Введите параметр прозрачности (от 0 до 1)\n"
            "Чем ближе к 0, тем больше изображение будет похоже на первое фото",
        )

        return float(res[0])
