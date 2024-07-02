import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale

# 初始化Tkinter窗口
root = tk.Tk()
root.title("Video Blurring Control")

# 全局变量，保存滑块的值
blur_level = 1


# 更新滑块的值
def update_blur_level(val):
    global blur_level
    blur_level = int(val)


# 创建滑块
scale = Scale(
    root,
    from_=1,
    to=20,
    orient=tk.HORIZONTAL,
    label="Blur Level",
    command=update_blur_level,
)
scale.pack()

# 打开视频捕捉
cap = cv2.VideoCapture(0)


def process_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # 模糊处理
    ksize = blur_level * 2 + 1
    blurred_frame = cv2.blur(frame, (ksize, ksize))

    # 显示处理后的视频
    cv2.imshow("Blurred Video", blurred_frame)

    # 每5毫秒调用一次process_frame
    root.after(5, process_frame)


# 启动视频处理
process_frame()

# 启动Tkinter主循环
root.mainloop()

# 释放视频捕捉
cap.release()
cv2.destroyAllWindows()
