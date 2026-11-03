# 运行时镜像：python:3.12-slim + ffmpeg（视频帧探针与清洗阶段重建视频依赖它）
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只拷依赖清单，利用层缓存：改代码不必重装依赖
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY pyproject.toml README.md main.py ./
COPY pipeline ./pipeline
COPY tools ./tools
COPY src ./src
COPY config ./config

# 安装本项目，提供 lerobot-data-quality 命令
RUN python -m pip install --no-deps .

# 约定：源数据集只读挂到 /data，产物写到 /outputs
RUN mkdir -p /data /outputs
VOLUME ["/data", "/outputs"]

ENTRYPOINT ["lerobot-data-quality"]
CMD ["--help"]
