define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make clean                  remove intermediate files

    make build-pytorch          make jupyter pytorch docker image
    make start-jupyter          launch jupyter server inside pytorch docker container
endef
export PROJECT_HELP_MSG

help:
	echo "$$PROJECT_HELP_MSG" | less

build-pytorch: pytorch-base
	docker build -t masalvar/pytorch .

pytorch-base: pytorch
	docker build -t masalvar/pytorch_base pytorch

pytorch:
    git clone https://github.com/pytorch/pytorch.git --branch v0.2.0

start-jupyter:
	nvidia-docker run --rm -ti --ipc=host -v /home/mat/repos:/workspace/repos -p 9999:9999 masalvar/pytorch jupyter notebook --allow-root

clean:
	rm -rf pytorch

.PHONY: help pytorch-base build-pytorch start-jupter clean
