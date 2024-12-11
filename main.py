import gradio as gr
from interface import *
from pathlib import Path


def various_operations(sample_image, file):

    if file is None:
        raise gr.Error("可恶，模型文件没选中啊💥!", duration=5)

    if Path(file).suffix != ".pt":
        raise gr.Error("可恶，模型文件后缀不是.pt啊💥!", duration=5)

    App.model = YOLO(file)

    gallery = tuple(
        zip(
            compose(sample_image),  # 保存processed_images
            (
                "原始图像",
                "直方图均衡化",
                "对比度拉伸",
                "自适应伽马校正",
                "图像的直方图",
                "均值滤波",
                "中值滤波",
                "双边滤波",
                "色相通道",
                "强度通道",
                "饱和度通道",
                "阈值分割",
                "大津算法分割",
                "YOLOv5检测(原始图像)",
                "YOLOv5检测(均值滤波后)",
                "YOLOv5检测(中值滤波后)",
                "YOLOv5检测(双边滤波后)",
            ),
        )
    )

    return gallery


demo = gr.Interface(
    fn=various_operations,
    inputs=["image", "file"],
    outputs=gr.Gallery(label="处理后的结果"),
    flagging_mode="never",
)


demo.launch()
