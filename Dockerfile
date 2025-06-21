FROM python:3.11-slim

# Create and switch to non-root user
RUN useradd -m -u 1000 user
USER user

# Environment
ENV HOME=/home/user
WORKDIR $HOME/app

# Install Python deps
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Create writable cache folders
RUN mkdir -p $HOME/app/cache

# Copy all source code
COPY --chown=user . .

# Run the app
CMD ["python", "app.py"]
