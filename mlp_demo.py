import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from mlp import MLP_NeuralNet

# 创建主窗口
root = tk.Tk()
root.title("Drawing App")

# 创建画布
canvas_size = 280
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.pack()

# 定义每个小格子的大小和颜色
cell_size = 10
cell_color = "black"

# 创建一个二维数组表示网格状态
grid = np.zeros((canvas_size // cell_size, canvas_size // cell_size), dtype=np.uint8)

# 创建一个画布图像
image = Image.new("L", (canvas_size, canvas_size), 255)
draw = ImageDraw.Draw(image)

# 创建MLP模型
mlp_model = MLP_NeuralNet()
mlp_model.load_param()

# 定义鼠标拖动的事件处理函数
def paint(event):
    x, y = event.x, event.y
    col = x // cell_size
    row = y // cell_size

    if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
        canvas.create_rectangle(col * cell_size, row * cell_size,
                                (col + 1) * cell_size, (row + 1) * cell_size,
                                fill=cell_color, outline="")

        # 更新网格状态
        grid[row, col] = 1

# 定义清屏按钮的点击事件处理函数
def clear_canvas():
    canvas.delete("all")
    # 清除网格状态
    grid.fill(0)
    # 清除画布图像
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=0)

# 定义鼠标释放的事件处理函数
def release(event):
    pass

# 定义将画布内容转化为输入的函数
def convert_and_predict():
    # 将画布内容转化为28x28的灰度图像
    img = image.resize((28, 28), Image.ANTIALIAS)
    img = img.convert("L")

    # 将灰度图像转换为784长度的输入
    input_data = grid.flatten()
    #input_data = input_data.reshape(1, -1)
    #print(input_data)

    # 调用MLP模型进行预测
    prediction = mlp_model.forward(input_data)
    predicted_label = np.argmax(prediction)
    print("Predicted Label:", predicted_label)

    # 0.1秒后再次调用该函数
    root.after(100, convert_and_predict)

# 创建清屏按钮
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

# 将画布与事件处理函数绑定
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", release)

# 启动主循环
root.after(100, convert_and_predict)  # 初始触发一次
root.mainloop()
