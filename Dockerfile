FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .

COPY uv_unwrapper/ ./uv_unwrapper/
COPY texture_baker/ ./texture_baker/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8080"]
