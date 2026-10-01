# make reproduce : rebuild the release bundle (results/release/) from the data on disk
# make test      : run regression tests (dates, brackets, units, fees, leakage, frozen trades)
PY?=python3
reproduce:
	$(PY) scripts/build_release.py
test:
	$(PY) -m pytest -q tests
