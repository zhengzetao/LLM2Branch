# LLM2Branch

Code for **Rethinking Large Language Model-Based Optimization for Mixed-Integer Linear Programming**.

LLM2Branch queries an LLM for native SCIP branching-rule recommendations during branch-and-bound. It includes initial/dynamic prompts, asynchronous requests, executable-rule guards and persistent hierarchical experience with UCB reuse. It selects rules rather than directly selecting branching variables.

Supported families: Set Covering (setcover), Combinatorial Auctions (cauctions), Capacitated Facility Location (facilities), Maximum Independent Set (indset).

## Installation

See [INSTALL.md](INSTALL.md). Validated server stack: Linux amd64, Python3.8, SCIP6.0.1, SoPlex4.0.1 and the modified ds4dm PySCIPOpt2.1.5 lineage. The implementation uses custom solver-interface methods; a current standard SCIP/PySCIPOpt installation is not a supported drop-in replacement.

The core path does not require TensorFlow, SVMrank, pretrained competitor models or a GPU.

## Quick start

Generate instances (100 at each of three transfer scales; run in a fresh output directory):

    python 01_generate_instances.py cauctions

Check the CLI and run one fixed-rule baseline without API credentials:

    python 07_llm-evaluate.py --help
    python 07_llm-evaluate.py cauctions --size small --limit 1 --baseline-only --time-limit 60

Configure a Chat Completions-compatible API:

    export LLM_API_KEY='YOUR_API_KEY'
    export LLM_API_URL='https://api.example.com/v1/chat/completions'
    export LLM_MODEL_NAME='your-model-name'
    python 07_llm-evaluate.py cauctions --size small --limit 1 --time-limit 120

Alternatively copy llm_config.example.py to llm_config.py and edit locally. Environment variables override the default profile. Never commit credentials. Provider-specific payload options in llm_tools/llm_agent.py may need adjustment for your API.

Replace cauctions with setcover, facilities or indset. Scale names are small, medium, big. The --limit argument selects the first N instances. The 06_llm-evaluate.py entry forwards to 07.

## Configuration and outputs

- experiment_config.py: instance paths, default budget, seeds and policies. The shipped internal comparator is fixed relpscost, not an untouched SCIP-default run.
- llm_settings.py: API settings.
- experience_index.py: experience hierarchy, reward calculation and UCB. Inspect ExperienceConfig for settings.
- utilities.py: solver parameters, state features, switching and query scheduling.
- experiment_results.py: metrics and result formatting.
- llm_tools/: client, prompts, parser and rule guards.

Runs create CSVs in results/, detailed experiment_results/ and persistent experience_indexes/. Reusing an index changes initial policy knowledge; use fresh indices for cold-start comparisons. Outputs and datasets are ignored by Git.

## Scope and reproducibility

This release contains the LLM2Branch core. Server diagnostics, datasets, checkpoints and output archives are excluded. Standalone RuleScheduler/FQI/TCN experiments are a separate project.

Execution checks do not demonstrate acceleration. Report solver time, LLM overhead and end-to-end time separately. API model, solver build and experience state affect outcomes.

## License and attribution

Generators and parts of the solver utilities derive from [Learn2Branch](https://github.com/ds4dm/learn2branch). Preserve its MIT notice in [LICENSE](LICENSE). Solver distributions have their own licenses.

## Release checks

The release was checked in the existing SCIP server environment: CLI help, a fresh-output fixed-rule baseline run, native solves of small generated instances from all four families, and an offline mocked LLM-client/configuration check. A live API call and a clean-from-source solver installation were not performed for this release.
