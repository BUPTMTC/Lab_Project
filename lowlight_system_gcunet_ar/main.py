# -*- coding:utf-8 -*-
import os

import PySide2

from form import Form

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
from PySide2.QtWidgets import QApplication

path = r'./Jup'
if not os.path.exists(path):
    os.mkdir(path)
app = QApplication()
myform = Form(path)
if __name__ == '__main__':
    myform.show()
    app.exec_()
