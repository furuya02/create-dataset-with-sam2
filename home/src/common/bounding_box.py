# マウスで範囲指定する
# apt-get install -y python3.10-tkが必要

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BoundingBox:
    def __init__(self):
        self.x1 = -1
        self.x2 = -1
        self.y1 = -1
        self.y2 = -1
        self.image = None

    def __motion(self, event):
        if self.x1 != -1 and self.y1 != -1:
            self.x2 = event.xdata.astype("int16")
            self.y2 = event.ydata.astype("int16")
            self.__update_plot()

    def __update_plot(self):
        plt.clf()
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        ax = plt.gca()
        rect = patches.Rectangle(
            (self.x1, self.y1),
            self.x2 - self.x1,
            self.y2 - self.y1,
            angle=0.0,
            fill=False,
            edgecolor="#00FFFF",
        )
        ax.add_patch(rect)
        plt.draw()

    def __press(self, event):
        self.x1 = event.xdata.astype("int16")
        self.y1 = event.ydata.astype("int16")

    def __release(self, event):
        plt.clf()
        plt.close()

    def get_box(self, image_path):
        self.image = cv2.imread(image_path)
        self.x1 = -1
        self.x2 = -1
        self.y1 = -1
        self.y2 = -1
        plt.figure()
        plt.connect("motion_notify_event", self.__motion)
        plt.connect("button_press_event", self.__press)
        plt.connect("button_release_event", self.__release)
        self.ln_v = plt.axvline(0)
        self.ln_h = plt.axhline(0)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.show()
        return (self.x1, self.y1, self.x2, self.y2)
