# -*- coding:utf-8 -*-
import numpy as np
import os
from astropy.io import fits
from threading import Thread
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class Handler(FileSystemEventHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.src_path = None

    def on_created(self, event):
        # 检测写入事件
        # 在检测到文件写入时，文件可能没有完全写入完成，所以等到检测到下一个文件写入时，读取上一次的检测结果
        if os.path.splitext(event.src_path)[-1] != '.fits':
            return
        if self.src_path is None:
            self.src_path = event.src_path
            return
        input_data = fits.getdata(self.src_path).astype(np.float32) / 65532
        if not self.queue.empty():
            self.queue.get()
        self.queue.put(input_data)
        try:
            os.remove(self.src_path)
        except:
            print("delete failed")
        finally:
            self.src_path = event.src_path


class Producer(Thread):
    def __init__(self, path, queue):
        super().__init__()
        self.path = path
        self.queue = queue

    def run(self):
        self.watch_folder()

    def watch_folder(self):
        # watch dog 检测的固定写法 详情可查看watch dog 文档
        event_handler = Handler(self.queue)
        observer = Observer()
        observer.schedule(event_handler, path=self.path, recursive=False)
        observer.start()
