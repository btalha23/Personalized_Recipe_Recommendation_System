#!/bin/bash

# Run using the below command
# bash setup_vm.sh

echo "Downloading anaconda..."
mkdir -p ../software && cd ~/software && \
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

echo "Running anaconda script..."
bash ~/software/Anaconda3-2024.06-1-Linux-x86_64.sh

echo "Removing anaconda script..."
rm ~/software/Anaconda3-2024.06-1-Linux-x86_64.sh

echo "Deactivating conda at startup..."
conda config --set auto_activate_base false

echo "Running sudo apt-get update..."
sudo apt update

echo "Installing Docker..."
sudo apt install docker.io

echo "Docker without sudo setup..."
sudo groupadd docker
sudo gpasswd -a ${USER} docker
sudo service docker restart

echo "Installing docker-compose..."
cd ~/software && wget https://github.com/docker/compose/releases/download/v2.29.2/docker-compose-linux-x86_64 -O docker-compose && \
sudo chmod +x docker-compose

echo "Setup .bashrc..."
echo '' >> ~/.bashrc
echo 'export PATH=${HOME}/software:${PATH}' >> ~/.bashrc
eval "$(cat ~/.bashrc | tail -n +10)"

echo "docker-compose version..."
docker-compose --version

echo "Activate conda environment..."
conda activate

echo "Making sure that python 3.11 is exploited in this project..."
conda install -c conda-forge python=3.11

echo "Installing necessary packages..."

pip install pipenv nltk mlflow prefect boto3 evidently pyarrow psycopg psycopg_binary pytest isort black pylint deepdiff