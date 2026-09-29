PYTHON ?= python3
SEEDS ?= 30
NOTEBOOK_OUTPUT_DIR ?= /tmp/tech-challenge-notebooks
export MLFLOW_TRACKING_URI ?= sqlite:///$(CURDIR)/mlflow.db

.PHONY: install data data-uci data-kaggle validate prepare train-policies evaluate quality-gate golden-set consolidate-policy approve-policy verify-mlflow mlflow-ui register-model data_pipeline eda notebooks eda-summary recommend api docker-build infra-fmt infra-validate infra-plan test lint clean
.NOTPARALLEL: notebooks

install:
	$(PYTHON) -m pip install -e '.[dev]'

data:
	$(PYTHON) scripts/download_data.py --source uci

data-kaggle:
	$(PYTHON) scripts/download_data.py --source kaggle

data-uci:
	$(PYTHON) scripts/download_data.py --source uci

validate:
	$(PYTHON) scripts/validate_data.py

prepare:
	$(PYTHON) scripts/prepare_data.py

train-policies:
	$(PYTHON) scripts/train_phase3_4.py

evaluate:
	$(PYTHON) scripts/evaluate_phase5.py --seeds $(SEEDS)

quality-gate:
	$(PYTHON) scripts/quality_gate.py

golden-set:
	$(PYTHON) scripts/build_golden_set.py

consolidate-policy:
	$(PYTHON) scripts/consolidate_policy.py

approve-policy:
	$(PYTHON) scripts/approve_policy.py --version "$(VERSION)" --approved-by "$(APPROVED_BY)"

verify-mlflow:
	$(PYTHON) scripts/verify_mlflow.py --minimum-seeds $(SEEDS)

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri "$(MLFLOW_TRACKING_URI)" --host 127.0.0.1 --port 5000

register-model:
	$(PYTHON) scripts/register_model.py

data_pipeline:
	$(MAKE) install
	$(MAKE) data
	$(MAKE) validate
	$(MAKE) prepare
	$(MAKE) train-policies
	$(MAKE) evaluate
	$(MAKE) golden-set

eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_preparation.ipynb

notebooks: data validate prepare train-policies evaluate golden-set
	mkdir -p "$(NOTEBOOK_OUTPUT_DIR)"
	jupyter nbconvert --to notebook --execute --output-dir="$(NOTEBOOK_OUTPUT_DIR)" \
		notebooks/01_eda_preparation.ipynb \
		notebooks/02_preparation_no_leakage.ipynb \
		notebooks/03_04_reward_models_policies.ipynb \
		notebooks/05_06_evaluation_golden_set.ipynb

eda-summary:
	$(PYTHON) scripts/run_eda.py

recommend:
	$(PYTHON) scripts/recommend.py

api:
	$(PYTHON) -m uvicorn adaptive_offers.api:app --app-dir src --host 127.0.0.1 --port 8000

docker-build:
	docker build -t adaptive-offers:local .

infra-fmt:
	terraform -chdir=infra/terraform fmt -recursive

infra-validate:
	terraform -chdir=infra/terraform init -backend=false -input=false
	terraform -chdir=infra/terraform validate

infra-plan:
	terraform -chdir=infra/terraform plan

test:
	pytest -q

lint:
	ruff check src scripts tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
