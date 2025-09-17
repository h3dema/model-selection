# model-selection

The code in this repository is used to generate synthetic models and datasets to simulate model selection scenarios.


## Install requirements

```
cd model-selection/
pyenv virtualenv 3.11 model-selection
pyenv activate model-selection
pip install -r requirements.txt
sudo apt-get install glpk-utils
```

`glpk-utils` is needed to run the optimization solver.
