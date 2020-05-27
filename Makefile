
VENV:=. .pyenv/bin/activate

.pyenv:
	virtualenv -p python3 .pyenv

requirements: .pyenv
	git submodule update --init
	. .pyenv/bin/activate && \
	pip install -e vendor/transformers && \
	pip install -e vendor/wandb-client && \
	pip install -e vendor/pytorch-lightning && \
	pip install -r requirements.txt

data/reddit-sarcasm:
	mkdir -p data/reddit-sarcasm

data/glue:
	mkdir -p data/glue

jupyter:
	${VENV} && cd notebooks && jupyter notebook
