FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag ./rag
COPY api.py app.py ./

RUN useradd --create-home appuser && mkdir -p /app/data && chown appuser /app/data
USER appuser

EXPOSE 8000 8501
# Default: the API. docker-compose also starts the Streamlit UI from this image.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
