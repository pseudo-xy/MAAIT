# MAAIT simulator — web console in one container.
# Build:  docker build -t maait .
# Run:    docker run -p 8000:8000 -e MAAIT_ADMIN_PASSCODE=yourcode maait
#         then open http://localhost:8000
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir "pydantic>=2.6" "numpy>=1.26" "scipy>=1.11" \
        "fastapi>=0.110" "uvicorn[standard]>=0.29"
COPY src ./src

# results CSVs, recordings and the saved scenario go to /data
# (attach a persistent disk there to keep them across restarts)
ENV PYTHONPATH=/app/src \
    MAAIT_DATA_DIR=/data \
    PORT=8000
RUN mkdir -p /data
EXPOSE 8000

# hosting services set $PORT; 8000 for a local docker run
CMD ["sh", "-c", "uvicorn reschu.web.server:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
