# Reproduce the paper's numbers in one container:
#   docker build -t myna-repro . && docker run --rm myna-repro
# It runs the aggregation over the committed per-question records and re-derives every table figure
# that the paper quotes, without needing a GPU.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /myna
COPY . /myna

RUN pip install --no-cache-dir uv && uv venv /myna/.venv && \
    uv pip install --python /myna/.venv/bin/python numpy pytest

CMD ["/myna/.venv/bin/python", "-m", "bench.paper_numbers"]
