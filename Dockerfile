FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY pipelines/ pipelines/
COPY monitoring/ monitoring/
COPY docker/ /docker/
COPY data.csv data.csv

RUN chmod +x /docker/entrypoint.sh /docker/start.sh

ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg

EXPOSE 8000 8501

CMD ["/docker/start.sh"]
