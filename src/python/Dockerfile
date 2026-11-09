# Python container: trains a logistic regression on the Titanic data.
FROM python:3.11-slim

WORKDIR /app

# Install pinned packages first so this layer is cached between code edits
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_model.py .

# The data is NOT baked into the image. Mount it at run time:
#   docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
ENV DATA_DIR=/app/data

CMD ["python", "titanic_model.py"]
