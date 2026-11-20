ARG PYTHON_IMAGE=python:3.8.20-slim-bookworm@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f
FROM ${PYTHON_IMAGE} AS toolchain
ARG DEBIAN_MIRROR=https://deb.debian.org
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCIPOPTDIR=/opt/scip LD_LIBRARY_PATH=/opt/scip/lib
RUN sed -i "s|http://deb.debian.org|$DEBIAN_MIRROR|g; s|http://security.debian.org|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake patch \
      libgmp-dev zlib1g-dev libreadline-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

FROM toolchain AS python-deps
ARG PIP_INDEX_URL=https://pypi.org/simple
WORKDIR /build
COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir pip==24.3.1 setuptools==68.2.2 wheel==0.45.1 \
 && python -m pip install --no-cache-dir Cython==0.29.37 -r requirements-runtime.txt

FROM python-deps AS build
ARG JOBS=4
# User-supplied archives are never distributed in this repository.
COPY soplex-4.0.1.tgz scip-6.0.1.tgz vanillafullstrong.patch ./
RUN mkdir soplex scip \
 && tar -xzf soplex-4.0.1.tgz -C soplex --strip-components=1 \
 && tar -xzf scip-6.0.1.tgz -C scip --strip-components=1 \
 && if [ ! -f soplex/src/soplex/git_hash.cpp ]; then \
      printf '%s\n' '#define SPX_GITHASH "release-401"' > soplex/src/soplex/git_hash.cpp; fi \
 && if [ ! -f scip/src/scip/githash.c ]; then \
      printf '%s\n' '#define SCIP_GITHASH "v601"' > scip/src/scip/githash.c; fi
RUN cmake -S soplex -B soplex/build -DCMAKE_INSTALL_PREFIX=/opt/scip \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=11 \
 && cmake --build soplex/build --parallel "$JOBS" \
 && cmake --install soplex/build
RUN cd scip \
 && patch -p1 < ../vanillafullstrong.patch \
 && cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/scip \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=11 \
      -DSOPLEX_DIR=/opt/scip -DZIMPL=OFF -DIPOPT=OFF \
 && cmake --build build --parallel "$JOBS" \
 && cmake --install build
# Immutable revision of ds4dm's ml-branching binding.
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://api.github.com/repos/ds4dm/PySCIPOpt/tarball/9b88eddcfe99208e737236d39ccd6c9dd1a9bec4', 'pyscipopt.tgz')" \
 && mkdir pyscipopt \
 && tar -xzf pyscipopt.tgz -C pyscipopt --strip-components=1 \
 && cd pyscipopt \
 && python -m pip install --no-cache-dir --no-build-isolation --no-deps .
RUN mkdir -p /opt/third-party-notices \
 && find /build/scip /build/soplex -maxdepth 1 -type f \
      \( -iname '*license*' -o -iname '*copying*' \) \
      -exec cp --parents '{}' /opt/third-party-notices/ \; \
 && cp /build/pyscipopt/LICENSE /opt/third-party-notices/PySCIPOpt-LICENSE

FROM ${PYTHON_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
ARG DEBIAN_MIRROR=https://deb.debian.org
RUN sed -i "s|http://deb.debian.org|$DEBIAN_MIRROR|g; s|http://security.debian.org|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends libgmp10 zlib1g libreadline8 libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local /usr/local
COPY --from=build /opt/scip /opt/scip
COPY --from=build /opt/third-party-notices /opt/third-party-notices
COPY verify_runtime.py /usr/local/bin/verify_runtime.py
ENV SCIPOPTDIR=/opt/scip LD_LIBRARY_PATH=/opt/scip/lib \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /workspace
RUN python /usr/local/bin/verify_runtime.py
CMD ["python", "/usr/local/bin/verify_runtime.py"]
