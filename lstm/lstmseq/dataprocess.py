import cv2
import pandas as pd

"""作用：
输入视频文件，输出csv文件，csv文件中一行为一个数字，表示视频帧的总像素值
"""


def calculate_frame_pixels(frame):
    # 计算图像的像素总和
    return frame.sum()


def main(_video_path, _output_csv):
    # 打开视频文件
    cap = cv2.VideoCapture(_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建一个空的 DataFrame 来存储像素总和
    data = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    # 逐帧处理视频
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame {}".format(i))
            break

        # 计算帧的像素总和
        pixel_sum = calculate_frame_pixels(frame)

        # 将像素总和添加到列表中
        data.append(pixel_sum)

    # 将数据保存到 CSV 文件中
    with open(_output_csv, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

    # 关闭视频流
    cap.release()



if __name__ == "__main__":
    # video_path = r'D:\pythonProject\laser-dec\data\laserdata0426-3s\de\de13.mp4'  # 视频文件路径
    # video_path = r'D:\pythonProject\laser-dec\data\preprocess\dusiming.mp4'  # 视频文件路径
    # video_path = r'D:\pythonProject\laser-dec\data\preprocess\liqiaoyun.mp4'  # 视频文件路径
    # video_path = r'D:\pythonProject\laser-dec\data\preprocess\du\de1.mp4'  # 视频文件路径

    # video_path = r'D:\pythonProject\laser-dec\data\preprocess\data0426\test.mp4'  # 视频文件路径
    video_path = r'D:\pythonProject\laser-dec\data\preprocess\wang.mp4'  # 视频文件路径
    output_csv = "video_pixels_wang.csv"  # 输出 CSV 文件路径
    main(video_path, output_csv)