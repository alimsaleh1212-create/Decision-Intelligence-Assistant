# Stage 1: install dependencies
# Use the official Python slim image so the layer is small and auditable.
FROM python:3.13-slim AS builder

# Install uv into the builder stage only
RUN pip install --no-cache-dir uv==0.11.3

WORKDIR /build

# Copy dependency manifests first so Docker caches the install layer
# separately from application code.
COPY pyproject.toml .

# Sync production deps only (no dev tools in the final image)
RUN uv sync --no-dev --no-cache

# Stage 2: runtime image
FROM python:3.13-slim AS runtime

# Non-root user — principle of least privilege
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the installed venv from builder
COPY --from=builder /build/.venv /app/.venv

# Put the venv on PATH so all executables are found without activation
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/

# Log directory (mounted as a named volume in docker-compose)
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

# Model directory (mounted at runtime)
RUN mkdir -p /app/models && chown appuser:appuser /app/models

USER appuser

EXPOSE 8000

# Use uvicorn directly — no shell wrapper needed
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
