# 快速开始

```powershell
git clone https://github.com/cqut-handsomeboy/class-design.git class-design
cd class-design
# 进入虚拟环境（如果需要）
poetry install # 安装依赖
python interface.py --image_path=/the/path/to/your/image --model_file_path=/the/path/to/your/model/file # 使用接口验证
python main.py # Gradio界面
```