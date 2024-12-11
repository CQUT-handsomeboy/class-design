import cv2
import fire.fire_import_test
import numpy as np
import fire

from matplotlib import pyplot as plt
from pathlib import Path

def open_image(file_path):
    """
    打开图像文件并返回图像对象。
    :param file_path: 图像文件路径
    :return: 图像对象
    """
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"无法打开文件：{file_path}")
    return image


def save_image(image, file_path):
    """
    保存图像文件。
    :param image: 图像对象
    :param file_path: 保存路径
    """
    cv2.imwrite(file_path, image)


def image_statistics(image):
    """
    统计图像的直方图并绘制。
    :param image: 图像对象
    """
    if len(image.shape) == 3:  # 彩色图像
        colors = ("b", "g", "r")
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title("Color histogram")
    else:  # 灰度图像
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color="black")
        plt.title("Grey histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Number of pixels")
    plt.show()


def enhance_image(image, method):
    """
    图像增强处理。
    :param image: 图像对象
    :param method: 增强方法: 'hist_equal' (直方图均衡化), 'contrast_stretch' (对比度展宽), 或 'adaptive_gamma' (动态伽马调整)
    :return: 增强后的图像
    """
    cv2.putText(
        image,
        f"Method: {method}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    if method == "hist_equal":
        if len(image.shape) == 3:  # 彩色图像
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:  # 灰度图像
            return cv2.equalizeHist(image)

    elif method == "contrast_stretch":
        in_min, in_max = np.percentile(image, (2, 98))
        out_min, out_max = 0, 255
        return cv2.normalize(
            image, None, alpha=out_min, beta=out_max, norm_type=cv2.NORM_MINMAX
        )

    elif method == "adaptive_gamma":
        gamma = 1.5  # 可动态调整
        look_up_table = np.array(
            [((i / 255.0) ** gamma) * 255 for i in np.arange(256)]
        ).astype("uint8")
        return cv2.LUT(image, look_up_table)
    else:
        raise ValueError(f"未知方法: {method}")


def spatial_filtering(image, method):
    """
    应用空间滤波算法。
    :param image: 图像对象
    :param method: 滤波方法: 'mean' (均值滤波), 'median' (中值滤波), 'bilateral' (边界保持滤波)
    :return: 滤波后的图像
    """
    cv2.putText(
        image,
        f"Method: {method}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    if method == "mean":
        return cv2.blur(image, (5, 5))

    elif method == "median":
        return cv2.medianBlur(image, 5)

    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)

    else:
        raise ValueError(f"未知方法: {method}")


def rgb_to_his(image):
    """
    将RGB图像转换为HIS空间，并显示其分量图。
    :param image: RGB图像对象
    :return: H、I、S分量图
    """
    cv2.putText(
        image,
        f"Method: rgb_to_his",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v


def segment_image(image, method="threshold", **params):
    """
    图像分割功能。
    :param image: 图像对象
    :param method: 分割方法: 'threshold' (阈值分割), 'otsu' (大津法)
    :param params: 方法参数
    :return: 分割后的图像
    """
    cv2.putText(
        image,
        f"Method: {method}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    if method == "threshold":
        thresh_value = params.get("thresh_value", 127)
        max_value = params.get("max_value", 255)
        _, segmented = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_BINARY)
        return segmented

    elif method == "otsu":
        _, segmented = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return segmented

    else:
        raise ValueError(f"未知方法: {method}")


def show(*images):
    for i, image in enumerate(images):
        cv2.imshow(str(i), image)
    cv2.waitKey(0)


def main(image_path:str):
    assert Path(image_path).exists(),"图片文件不存在"
    sample_image = cv2.resize(cv2.imread(image_path), (400, 525)) # 缩放的逻辑需要调整
    show(
        sample_image.copy(),
        enhance_image(sample_image.copy(), "hist_equal"),
        enhance_image(sample_image.copy(), "contrast_stretch"),
        enhance_image(sample_image.copy(), "adaptive_gamma"),
        spatial_filtering(sample_image.copy(), "mean"),
        spatial_filtering(sample_image.copy(), "median"),
        spatial_filtering(sample_image.copy(), "bilateral"),
        *rgb_to_his(sample_image.copy()),
        segment_image(sample_image.copy(), "threshold"),
    )


if __name__ == "__main__":
    fire.Fire(main)
