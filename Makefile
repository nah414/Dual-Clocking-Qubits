.PHONY: setup test run-ion run-sc run-nmr run-nmr-grad

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev]

test:
	pytest -q

run-sc:
	python scripts/demo_sc.py

run-ion:
        python scripts/demo_ion.py

run-nmr:
        python -m examples.run_nmr

run-nmr-grad:
        python -m examples.run_nmr_grad
