FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV TEAM_ID_STR=TEAM_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["./run.sh"]
