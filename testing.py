import cv2
import numpy as np
import math
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


# 视频处理函数
def pixelate_half_frame(frame, gridSize):
    height, width, channels = frame.shape
    output = np.copy(frame)
    for y in range(0, height, gridSize):
        for x in range(0, width, gridSize):
            rect = frame[y : y + gridSize, x : x + gridSize]
            (b, g, r) = cv2.mean(rect)[:3]
            cv2.rectangle(output, (x, y), (x + gridSize, y + gridSize), (b, g, r), -1)
    return output


def initialize_accumulator_grid(rows, cols, channels):
    return np.zeros((rows, cols, channels + 1), dtype=np.float64)


def update_accumulator_grid(accumulator, frame, x, y, gridSize):
    block = frame[y : y + gridSize, x : x + gridSize]
    mean_color = cv2.mean(block)[:3]
    accumulator[y // gridSize, x // gridSize, :3] += mean_color
    accumulator[y // gridSize, x // gridSize, 3] += 1
    return accumulator


def calculate_average_from_accumulator(accumulator):
    count = accumulator[..., 3]
    count[count == 0] = 1  # Avoid division by zero
    averages = accumulator[..., :3] / count[..., np.newaxis]
    return averages


# 打开视频捕捉
cap = cv2.VideoCapture("TajMahalDaytoNight.mp4")
if not cap.isOpened():
    print("Error opening video file")
    exit(0)

ret, sample_frame = cap.read()
if not ret:
    print("No frames to read")
    exit(0)

height, width, channels = sample_frame.shape
gridSize = 10
rows = math.ceil(height // gridSize)
cols = math.ceil(width // gridSize)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
gridStep = max(1, math.ceil((rows * cols) // total_frames))

accumulator_grid = initialize_accumulator_grid(rows, cols, channels)
final_image = np.zeros((rows * gridSize, cols * gridSize, 3), dtype=np.uint8)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start of video

frame_idx = 0
block_idx = 0
frame_count = gridStep

total_blocks = rows * cols

if total_frames > total_blocks:
    frames_per_block = total_frames // total_blocks
else:
    blocks_per_frame = total_blocks // total_frames
    remaining_blocks = total_blocks % total_frames


def process_frame():
    global frame_idx, block_idx
    ret, frame = cap.read()
    if not ret:
        return

    if total_frames > total_blocks:
        block_idx = frame_idx // frames_per_block
        if block_idx >= total_blocks:
            return
        block_y = (block_idx % rows) * gridSize
        block_x = (block_idx // rows) * gridSize
        update_accumulator_grid(accumulator_grid, frame, block_x, block_y, gridSize)
        if (
            frame_idx % frames_per_block == frames_per_block - 1
            or frame_idx == total_frames - 1
        ):
            averages = calculate_average_from_accumulator(
                accumulator_grid[block_idx % rows, block_idx // rows]
            )
            color = tuple(int(c) for c in averages)
            cv2.rectangle(
                final_image,
                (block_x, block_y),
                (block_x + gridSize, block_y + gridSize),
                color,
                -1,
            )
    else:
        start_block_idx = frame_idx * blocks_per_frame
        end_block_idx = start_block_idx + blocks_per_frame
        if frame_idx == total_frames - 1:
            end_block_idx += remaining_blocks
        for block_idx in range(start_block_idx, end_block_idx):
            if block_idx >= total_blocks:
                return
            block_y = (block_idx % rows) * gridSize
            block_x = (block_idx // rows) * gridSize
            update_accumulator_grid(accumulator_grid, frame, block_x, block_y, gridSize)
            averages = calculate_average_from_accumulator(
                accumulator_grid[block_idx % rows, block_idx // rows]
            )
            color = tuple(int(c) for c in averages)
            cv2.rectangle(
                final_image,
                (block_x, block_y),
                (block_x + gridSize, block_y + gridSize),
                color,
                -1,
            )

    frame_idx += 1

    pixelated_frame = pixelate_half_frame(frame, blur_level * 2)
    frame_resized = cv2.resize(frame, (width // 7, height // 7))
    pixelated_frame_resized = cv2.resize(pixelated_frame, (width // 7, height // 7))
    final_image_resized = cv2.resize(final_image, (width // 7, height // 7))
    cv2.imshow("Original Video", frame_resized)
    cv2.imshow("Constructing Image", final_image_resized)
    cv2.imshow("Pixelated Video", pixelated_frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        root.quit()

    root.after(5, process_frame)


# 启动视频处理
root.after(0, process_frame)

# 启动Tkinter主循环
root.mainloop()

# 释放资源
cap.release()
cv2.destroyAllWindows()
