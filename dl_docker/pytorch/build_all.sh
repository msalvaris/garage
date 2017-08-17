git clone https://github.com/pytorch/pytorch.git --branch v0.2.0
cd pytorch
docker build -t masalvar/pytorch_base .
cd ..
docker build -t masalvar/pytorch .

