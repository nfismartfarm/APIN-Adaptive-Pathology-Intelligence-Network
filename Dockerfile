FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/best_model.pt models/best_model.pt
COPY models/temperature.pt models/temperature.pt
COPY diagnosis/ diagnosis/
COPY templates/ templates/
COPY static/ static/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
