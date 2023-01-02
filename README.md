# Tree-Based-Risk-Prediction

Liên kết: https://docs.google.com/document/d/1Lx5sLDXH7LIhN2q01FCrucnWelzzCQPJPaC_8DiaklI/edit#

### Preriquirements

Before you begin, you must install Python 3 and set up the environment as described below:

If you wish, you can create env using Anaconda / Miniconda or Python Env. Below, I'll demonstrate how I create my environment using Python's venv command.

```
python3 -m venv <name_of_virtualenv>
```

For instance, the name of my environment is 'env':

```
python3 -m venv env
```

Activate `env`:

```
source env/bin/activate
```

Install the requirements library in the `requirements.txt` file:

```
pip install -r requirements.txt
```

List library we use in this project:

```
lifelines
numpy
pandas
sklearn
seaborn
matplotlib
pydotplus
graphviz
shap
more-itertools
streamlit
```

To run project:

```
cd app
streamline run app.py
```

# Note

Khi cài đặt streamlit và sử dụng, ở phiên bản python 3.9.7 có thể gặp lỗi protocols, bạn có thể khắc phục bằng cách nâng cấp lên phiên bản 3.10.
