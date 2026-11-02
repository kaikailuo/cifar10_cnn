# Agent instructions

Start by reading [HANDOFF.md](HANDOFF.md): current state, how to run the experiments on the GPU server, the rules (test set used once, configs frozen once run, no invented numbers), and how to write the report. Project overview and experiment table: [README.md](README.md).

- Run tests with `python -m pytest -q` and a pipeline check with `python -m src.run --smoke --jobs 4` after any code change.
- Ask the author before committing.
