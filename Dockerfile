# ===============================
# Stage 1 — Builder
# ===============================
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (cache optimization)
COPY requirements.txt .
COPY pyproject.toml .

# Upgrade pip
RUN pip install --upgrade pip

# Install build tool
RUN pip install build

# Install dependencies
RUN pip install -r requirements.txt

# Copy full source
COPY . .

# Build wheel
RUN python -m build


# ===============================
# Stage 2 — Runtime
# ===============================
FROM python:3.10-slim

WORKDIR /app

# Create non-root user (production practice)
RUN useradd -m appuser
USER appuser

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl /app/

# Install the built package
RUN pip install --upgrade pip && \
    pip install /app/*.whl

# Copy any runtime assets
COPY --from=builder /app/src /app/src
COPY --from=builder /app/scripts /app/scripts

# Expose inference port
EXPOSE 8000

# Default command (serving mode)
ENTRYPOINT ["python", "-m", "src.main"]

