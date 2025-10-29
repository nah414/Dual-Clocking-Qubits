.PHONY: run-nmr

run-nmr:
	python -m examples.run_nmr --mode fid --amp 1.0 --strength 0.5 --acq 2.56e-3
