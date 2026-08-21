# Oracle: build once, run as either a master or a worker.
#
#   docker build -t oracle:latest .
#   docker run --rm -p 8000:8000 -v /models:/models:ro oracle:latest \
#       oracle-engine-master --config /etc/oracle/single.toml --model /models/model.gguf

FROM debian:bookworm-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY include ./include
COPY src ./src
COPY apps ./apps
COPY tests ./tests
COPY third_party ./third_party
COPY configs ./configs

# Tests run at build time: an image that cannot pass its own suite should not
# be produced at all.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORACLE_BUILD_TESTS=ON \
    && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure \
    && cmake --install build --prefix /out

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 oracle

COPY --from=build /out/bin/ /usr/local/bin/
COPY --from=build /out/share/oracle/configs/ /etc/oracle/

# Models are mounted read-only at run time; they are far too large to bake in.
VOLUME ["/models"]
WORKDIR /var/lib/oracle
RUN chown oracle:oracle /var/lib/oracle
USER oracle

EXPOSE 8000/tcp 9200/tcp 9100/udp

HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT []
# Runs the whole model on this container.  Authentication is on by default, so
# supply ORACLE_API_KEYS (or override the command with --no-auth):
#   docker run -e ORACLE_API_KEYS="demo:$(openssl rand -hex 24)" ...
CMD ["oracle-engine-master", "--config", "/etc/oracle/single.toml", "--single"]
