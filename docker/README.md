# Rebuild the CPU solver environment

This directory publishes a Docker **build recipe**, not a prebuilt SCIP image.
It uses Python 3.8.20 and preserves SCIP 6.0.1, SoPlex 4.0.1, the vanillafullstrong patch,
and the ds4dm PySCIPOpt 2.1.5 custom APIs required by the released code.
It does not install an LLM SDK, GPU training frameworks, project code, API
settings, datasets, experiment outputs, checkpoints, or execution histories.

## Why the solver archives are supplied by the user

SCIP 6.0.1 and SoPlex 4.0.1 use the ZIB Academic License, rather than the
license of current SCIP releases. Obtain them from the maintainers and
review/accept their terms yourself. The legacy SCIP license limits eligible
users and imposes redistribution conditions including access tracking and
corresponding source. Do not assume a locally built image can be publicly
redistributed without satisfying those conditions.

- SCIP download: https://scipopt.org/download.php
- SoPlex download: https://soplex.zib.de/download.php
- Legacy SCIP license: https://scipopt.org/doc-6.0.2/html/LICENSE.php
- Upstream source tags: https://github.com/scipopt/scip/tree/v601 and
  https://github.com/scipopt/soplex/tree/release-401

Download the **source archives**, not a binary installer. Archives from these
exact source tags can also be supplied; the script normalizes their filenames.
Keep them outside the repository. No solver archive is uploaded here.

## Build on a Linux machine with Docker

From the repository root, using your locally obtained archives:

    bash docker/build.sh /absolute/path/scip-6.0.1.tgz /absolute/path/soplex-4.0.1.tgz

The default image tag is llm2branch:scip6-cpu. The script creates an isolated,
temporary build context with seven explicitly selected files, prints the two
archive SHA-256 hashes, builds for Linux amd64, and runs an offline solver check.
It never sends your project directory to Docker as the build context.

Numerical dependencies are pinned in requirements-runtime.txt;
PySCIPOpt is pinned to commit 9b88eddcfe99208e737236d39ccd6c9dd1a9bec4.
Docker needs network access to obtain its base image, Debian build packages,
Python wheels, and the public binding source. Optional build settings:

    JOBS=8 bash docker/build.sh /path/scip.tgz /path/soplex.tgz my-local-tag

The PYTHON_IMAGE environment variable can select a trusted mirror of the same
Python 3.8.20 Debian Bookworm image if Docker Hub is unavailable. The default
base image is pinned by digest, but Debian repositories are not immutable
snapshots; this recipe reconstructs software versions and interfaces, rather
than promising a byte-identical copy of the original server. The original
server uses Ubuntu 20.04 and Python 3.8.10; this image uses Debian Bookworm
and Python 3.8.20.

For a slow connection to the default registries, the validation server used
these optional mirror settings:

    PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.8.20-slim-bookworm@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f \
    DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    JOBS=8 bash docker/build.sh /path/scip.tgz /path/soplex.tgz

The Python image digest is the same as the default base. No private package
credentials should be passed through build arguments.


## Run the published code

The code and data are mounted at runtime and remain outside the image:

    docker run --rm --network none --user "$(id -u):$(id -g)" \
      -v "$PWD:/workspace" llm2branch:scip6-cpu \
      python 07_llm-evaluate.py --help

After extracting the data-v1 assets into the repository root:

    docker run --rm --network none --user "$(id -u):$(id -g)" \
      -v "$PWD:/workspace" llm2branch:scip6-cpu \
      python 07_llm-evaluate.py cauctions --size small --limit 1 \
      --baseline-only --time-limit 60

SCIP baseline evaluation needs no API. To use the LLM policy, the user must
provide their own runtime settings as described in the main README and enable
network access for that run. Credentials must remain outside the Docker build
context. Outputs are generated in the mounted checkout, not embedded in the image.

This runtime supports the currently published LLM2Branch evaluation and instance
generation code. It excludes the separate RuleScheduler training stack and
optional SVMrank competitor; those are not required for this project's baseline
and LLM branching evaluation.

## Validation

The verify_runtime.py script imports the dependencies, checks the custom binding
methods and vanillafullstrong plugin, then solves a synthetic binary problem to
optimality. The build fails if this check fails. This is an installation check,
not an experimental result or reproduction of the paper's performance claims.

The recipe was validated on Linux amd64 by building the solver and binding from
source, running the image self-check with networking disabled, generating and
solving tiny instances for all four families, and invoking the released baseline
CLI for each family. No live LLM API request was made. The image layers were also
checked for project configuration, experiment directories and credential leakage.
