FROM python:3.12-slim

WORKDIR /app

# Install dependencies first to leverage layer caching
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code AND the trained model artifact
COPY . .

# The API listens on port 8000
EXPOSE 8000

# Start the server. 0.0.0.0 is mandatory so the API is reachable
# from OUTSIDE the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
