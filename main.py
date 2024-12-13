import gradio as gr
from interface import *
from pathlib import Path


def various_operations(sample_image, file_path):

    if file_path is None:
        raise gr.Error("可恶，模型文件没选中啊💥!", duration=5)

    if Path(file_path).suffix != ".pt":
        raise gr.Error("可恶，模型文件后缀不是.pt啊💥!", duration=5)

    App.model = YOLO(file_path)

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
    gr.Info("日志文件：logs/app.log")

    return gallery


with gr.Blocks() as demo:
    sample_image = gr.Image(type="numpy", label="输入图片")
    model_file_path = gr.File(
        label="模型文件", file_types=[".pt"], type="filepath"
    )
    gallery = gr.Gallery(label="图像处理结果")
    submit_button = gr.Button("处理").click(
        fn=various_operations,
        inputs=[sample_image, model_file_path],
        outputs=gallery,
    )

demo.launch(server_port=8080)
