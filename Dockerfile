FROM python:3.11-slim

# Create & use non-root user
RUN useradd -m -u 1000 user
USER user

# Environment
ENV HOME=/home/user
WORKDIR $HOME/app

# Installation
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Caching
COPY --chown=user . .
RUN mkdir -p cache

CMD ["python", "app.py"]
