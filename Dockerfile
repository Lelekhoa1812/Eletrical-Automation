# --------- PopTech real-time cleaning pipeline ---------
FROM python:3.11-slim

# 1. system deps for scikit-learn wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source late to leverage Docker layer caching
COPY . .

# ensure logs appear instantly in HF Space dashboard
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
