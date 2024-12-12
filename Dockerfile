# 使用官方的Python基础镜像
FROM python:3.12-alpine

# 安装Poetry
RUN pip install poetry

# 设置工作目录
WORKDIR /app

# 将当前目录复制到镜像中
COPY . .

# 安装依赖项
RUN poetry install --no-dev

EXPOSE 8080

# 运行main.py
CMD ["python", "main.py"]
