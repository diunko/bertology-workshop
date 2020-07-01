
VENV:=. .pyenv/bin/activate

.pyenv:
	virtualenv -p python3 .pyenv

requirements: .pyenv
	git submodule update --init
	${VENV} && pip install -r requirements.txt

data/glue/_done:
	python -m utils.download_glue_data \
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

blackd:
	blackd --bind-port 9090
