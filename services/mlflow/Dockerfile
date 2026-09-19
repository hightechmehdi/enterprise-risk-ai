FROM python:3.12-slim

WORKDIR /home/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

CMD mlflow server -p $PORT \
    --host 0.0.0.0 \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --allowed-hosts "*"
