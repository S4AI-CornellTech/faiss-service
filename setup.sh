conda env create -f environment.yml
conda activate faiss-service

sudo apt update
sudo apt install civetweb libcivetweb-dev intel-mkl-full protobuf-compiler-grpc libssl-dev cmake libgrpc++-dev protobuf-compiler libspdlog-dev libgrpc-dev libgflags-dev libgtest-dev libc-ares-dev libprotobuf-dev

pip install gdown

gdown --id 1xFBQnltn_KtwSjE-aGIChgwxtyKroneJ -O triviaqa_encodings.npy

python create_synthetic_index.py --index-size 10m --dim 768

make cppclient

sudo mkdir -p /usr/lib/x86_64-linux-gnu/cmake/grpc
sudo cp grpc-config.cmake /usr/lib/x86_64-linux-gnu/cmake/grpc
mkdir build; cd build
cmake ..
make

cd ../

docker build -t faiss-server .