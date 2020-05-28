
VENV:=. .pyenv/bin/activate

.pyenv:
	virtualenv -p python3 .pyenv

requirements: .pyenv
	git submodule update --init
	${VENV} && \
		pip install -e vendor/transformers && \
		pip install -e vendor/wandb-client && \
		pip install -e vendor/pytorch-lightning && \
		pip install -r vendor/transformers/examples/requirements.txt && \
		pip install -r requirements.txt

data/reddit-sarcasm:
	mkdir -p data/reddit-sarcasm

data/glue/_done:
	python -m bertology.utils.download_glue_data \
		--data_dir data/glue
	touch data/glue/_done

data/glue: data/glue/_done

ver:
	python -V
	which python
	exit 1
	echo going on

jupyter:
	${VENV} && cd notebooks && jupyter notebook
