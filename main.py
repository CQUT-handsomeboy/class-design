import gradio as gr
from interface import *
from pathlib import Path


def various_operations(sample_image, file):

    if file is None:
        raise gr.Error("可恶，模型文件没选中啊💥!", duration=5)

    if Path(file).suffix != ".pt":
        raise gr.Error("可恶，模型文件后缀不是.pt啊💥!", duration=5)

    App.model = YOLO(file)

    return [
        (sample_image.copy(), "原始图像"),
        (enhance_image(sample_image.copy(), "hist_equal"), "直方图均衡化"),
        (enhance_image(sample_image.copy(), "contrast_stretch"), "对比度拉伸"),
        (enhance_image(sample_image.copy(), "adaptive_gamma"), "自适应伽马校正"),
        ((f1 := spatial_filtering(sample_image.copy(), "mean")), "均值滤波"),
        ((f2 := spatial_filtering(sample_image.copy(), "median")), "中值滤波"),
        ((f3 := spatial_filtering(sample_image.copy(), "bilateral")), "双边滤波"),
        (rgb_to_his(sample_image.copy())[0], "色相通道"),
        (rgb_to_his(sample_image.copy())[1], "强度通道"),
        (rgb_to_his(sample_image.copy())[2], "饱和度通道"),
        (segment_image(sample_image.copy(), "threshold"), "阈值分割"),
        (yolov5_detect(sample_image.copy()), "YOLOv5检测（原始图像）"),
        (yolov5_detect(f1.copy()), "YOLOv5检测（均值滤波后）"),
        (yolov5_detect(f2.copy()), "YOLOv5检测（中值滤波后）"),
        (yolov5_detect(f3.copy()), "YOLOv5检测（双边滤波后）"),
    ]


demo = gr.Interface(
    fn=various_operations,
    inputs=["image", "file"],
    outputs=gr.Gallery(label="处理后的结果"),
)


demo.launch(debug=True)
