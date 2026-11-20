# Installation

For an isolated CPU environment, use the [Docker build guide](docker/README.md).
The recipe builds the legacy solver from user-supplied sources, pins the custom
binding revision, and keeps code, data, credentials and outputs outside the image.


## Platform and Python

The server deployment uses Linux amd64, Python3.8, SCIP6.0.1, SoPlex4.0.1 and the modified ds4dm PySCIPOpt2.1.5 lineage. This legacy integration needs custom solver methods; importing a standard modern pyscipopt package alone is insufficient.

    conda create -n llm2branch python=3.8 pip
    conda activate llm2branch
    pip install -r requirements.txt
    pip install 'Cython<3'
    export SCIPOPTDIR=/opt/scip
    export LD_LIBRARY_PATH="$SCIPOPTDIR/lib:$LD_LIBRARY_PATH"

## Native solver

Obtain SoPlex4.0.1 and SCIP6.0.1 source distributions from their upstream maintainers under their licenses. Extract alongside the checkout; adjust paths as needed.

    cmake -S soplex-4.0.1 -B soplex-4.0.1/build -DCMAKE_INSTALL_PREFIX="$SCIPOPTDIR"
    cmake --build soplex-4.0.1/build -j4
    cmake --install soplex-4.0.1/build
    cd scip-6.0.1
    patch -p1 < ../LLM2Branch/scip_patch/vanillafullstrong.patch
    cd ..
    cmake -S scip-6.0.1 -B scip-6.0.1/build -DSOPLEX_DIR="$SCIPOPTDIR" -DCMAKE_INSTALL_PREFIX="$SCIPOPTDIR"
    cmake --build scip-6.0.1/build -j4
    cmake --install scip-6.0.1/build

Install the modified binding used by Learn2Branch:

    pip install 'git+https://github.com/ds4dm/PySCIPOpt.git@ml-branching'

This is the historical dependency branch, not an immutable revision lock. The included SCIP patch enables vanillafullstrong. The runtime additionally needs custom feature/state and native-rule execution APIs called in utilities.py and 07_llm-evaluate.py. Audit those methods when validating a binding build.

## Verify

From the repository checkout:

    python 07_llm-evaluate.py --help
    python 01_generate_instances.py cauctions
    python 07_llm-evaluate.py cauctions --size small --limit 1 --baseline-only --time-limit 60

Then configure the API as described in README.md. SVMrank is optional, loaded only for explicitly configured SVMrank competitor policies.

The Docker recipe has been built from source on Linux amd64 and checked offline with the released CLI and tiny instances from all four problem families. The manual native-install procedure above has not been independently validated on a new machine. These checks verify installation and interfaces, not benchmark performance or live LLM API behavior.
