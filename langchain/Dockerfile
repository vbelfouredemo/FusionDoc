FROM python:3.9-slim

WORKDIR /app

# Install minimal dependencies for web scraping
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    ca-certificates \
    && apt-get clean

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright with Chromium
RUN pip install playwright==1.41.1 && \
    playwright install chromium && \
    playwright install-deps chromium

COPY . .

CMD ["python", "app.py"]