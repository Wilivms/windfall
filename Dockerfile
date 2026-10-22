FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY windfall ./windfall
EXPOSE 8080
CMD ["uvicorn", "windfall.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
