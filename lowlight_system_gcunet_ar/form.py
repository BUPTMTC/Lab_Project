from queue import Queue

import PySide2
from PySide2.QtCore import QFile, Slot
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QWidget

from Qconsumer import Consumer
from Qproducer import Producer
from ui.ui_mainwindow import Ui_Form


class Form(QWidget):
    def __init__(self, path):
        super().__init__()
        qfile = QFile("./ui/form.ui")
        qfile.open(QFile.ReadOnly)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("低照度增强系统")
        self.ratioSlider = self.ui.ratioSlider
        self.ratioSlider.setValue(30)
        self.ui.rlabel.setText("Ratio:%.3f" % (self.ratioSlider.value() / self.ratioSlider.maximum()))
        self.inputQue = Queue(1)
        self.c = Consumer(self, self.inputQue)
        self.p = Producer(path, self.inputQue)

        self.ratioSlider.valueChanged.connect(self.valueChanged)
        self.c.processDone.OnProcessDone.connect(self.setPixel)

        self.p.daemon = True
        self.p.start()
        self.c.start()

    @Slot(int)
    def valueChanged(self, val):
        print(val)
        v = val / self.ratioSlider.maximum()
        self.ui.rlabel.setText("Ratio:%.3f" % v)
        self.c.set_thr(v)

    @Slot(object)
    def setPixel(self, img):
        # img.scaled(ui->label->size(), Qt::KeepAspectRatio);
        self.ui.label.setScaledContents(True)
        self.ui.label.setPixmap(img)

    def closeEvent(self, event: PySide2.QtGui.QCloseEvent) -> None:
        print("窗口关闭")
        self.c.requestInterruption()
        self.c.quit()
        self.c.wait()
        event.accept()
