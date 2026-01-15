sudo apt update
sudo apt install python3-pip -y
pip3 install mlflow pandas scikit-learn
python3 train.py
mlflow ui
