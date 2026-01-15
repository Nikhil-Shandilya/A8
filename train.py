import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
data={
"hours":[1,2,3,4],
"marks":[40,50,78,90]
}
df=pd.DataFrame(data)
X=df[['hours']]
y=df['marks']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=LinearRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)
mse=mean_squared_error(y_test,pred)
mlflow.start_run()
mlflow.log_param("model_type","LinearRegression")
mlflow.log_metric("mse",mse)
mlflow.sklearn.log_model(model,"model")
mlflow.end_run()
print("Model trained and logged by mlflow")

"""
sudo apt update
sudo apt install python3-pip -y
pip3 install mlflow pandas scikit-learn
python3 train.py
mlflow ui
find ~ -type f -name "activate"

find ~ -type f -name "activate"


/home/nick/myenv/bin/activate
cmd: /home/nick/myenv/bin/python clean.py

~/.venvs/myenv
cmd: $HOME/.venvs/myenv/bin/python clean.py

project/
  .venv/
  cmd: .venv/bin/python clean.py

which python
/home/nick/myenv/bin/python

python -c "import sys; print(sys.executable)"
/home/nick/myenv/bin/python


"""
