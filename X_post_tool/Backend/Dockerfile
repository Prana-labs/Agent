FROM python:3.12-slim

WORKDIR /app

# Copy requirements from either subfolder or direct folder context
COPY requirements.txt* X_post_tool/Backend/requirements.txt* ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set PYTHONPATH so all internal modules can be imported
ENV PYTHONPATH="/app:/app/X_post_tool/Backend:/app/Backend"

# Start uvicorn from the directory containing main.py
CMD ["sh", "-c", "if [ -f main.py ]; then uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}; elif [ -d X_post_tool/Backend ]; then cd X_post_tool/Backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}; else uvicorn Backend.main:app --host 0.0.0.0 --port ${PORT:-8000}; fi"]