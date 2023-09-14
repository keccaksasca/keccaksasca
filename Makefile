all: install-full 

install-base:
	python3 -m pip install -r requirements.txt

install-full: install-base
	python3 -m pip install -r requirements_repo.txt

.PHONY: install-base install-full all
