.PHONY: install train report app api reproduce clean

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:                ## Train + save the best-of-class model artifact
	$(PY) -m src.train

report:               ## Regenerate all chapter tables and figures into reports/
	$(PY) -m src.report

app:                  ## Launch the Streamlit app
	$(VENV)/bin/streamlit run app.py

api:                  ## Launch the FastAPI prediction service
	$(VENV)/bin/uvicorn backend.api:app --reload

reproduce: train report   ## Full reproduction: model + every chapter asset
	@echo "Reproduction complete. See models/ and reports/."

clean:
	rm -rf reports/*.pdf reports/*.csv reports/metrics_summary.json
	rm -f models/diabetes_model.joblib models/metrics.json
