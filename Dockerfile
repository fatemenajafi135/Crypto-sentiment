FROM python:3.11-slim

WORKDIR /app

RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ src/
COPY config.py .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
