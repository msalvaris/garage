define PROJECT_HELP_MSG

Usage:
    make help                   show this message
    make clean                  remove intermediate files

	make pytorch				make jupyter pytorch docker image
    make start-jupyter          launch jupyter server inside pytorch docker container

endef
export PROJECT_HELP_MSG

help:
	echo $$PROJECT_HELP_MSG | less

pytorch: pytorch-base
	docker build -t masalvar/pytorch .

pytorch-base:
	git clone https://github.com/pytorch/pytorch.git --branch v0.2.0
	docker build -t masalvar/pytorch_base pytorch

start-jupyter:
	nvidia-docker run --rm -ti --ipc=host -v /home/mat/repos:/workspace/repos -p 9999:9999 masalvar/pytorch jupyter notebook --allow-root

clean:
	rm -rf pytorch

.PHONY: help pytorch pytorch-base start-jupter clean