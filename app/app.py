from utils import c_index
import base64
import numpy as np
import pandas as pd
import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
from load_data import get_training_data
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))


X_train, y_train, X_val, y_val = get_training_data()
print('X_train')
print(X_train)
st.write("""
# DỰ ĐOÁN KHẢ NĂNG BỆNH NHÂN CÓ NGUY CƠ BỊ ĐỘT QUỴ
""")

classifier_name = st.sidebar.selectbox(
    'Lựa chọn mô hình phần lớp',
    ('', 'Decision Tree', 'Random Forest')
)


def st_pandas_to_csv_download_link(_df: pd.DataFrame, file_name: str = "test_data.csv"):
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Tải xuống sample data(CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)


def add_parameter(clf_name):
    params = dict()
    if clf_name == '':
        st.write("""
Theo Tổ chức Y tế Thế giới (WHO) đột quỵ là nguyên nhân gây Đột quỵ thứ 2 trên toàn cầu, chiếm khoảng 11% tổng số ca Đột quỵ[1]. Bệnh đột quỵ thường xảy ra đột ngột và khó có thể kiểm soát nên gây rất nhiều khó khăn cho người nhà bệnh nhân và các y bác sĩ. Chính vì vậy bọn em xin đề xuất một hệ thống “Dự đoán khả năng bệnh nhân có nguy cơ bị đột quỵ “ bằng các phương pháp học máy như cây quyết định (Decision Tree) [2] và rừng ngẫu nhiên (Random Forest) [3]. Chức năng chính của bọn em là sử dụng dữ liệu quá khứ của bệnh nhân bị đột quỵ để dự đoán liệu một bệnh nhân có khả năng bị đột quỵ hay không dựa trên các thông số đầu vào như giới tính, tuổi tác, các bệnh khác nhau và tình trạng hút thuốc. Mỗi hàng trong dữ liệu cung cấp thông tin liên quan về bệnh nhân.

Dữ liệu sử dụng để đưa ra dự đoán trong ứng dụng này có định dạng giống như sample data, và bạn có thể tải xuống sample data ở bên dưới.

""")
        df_sample_data = pd.read_csv("data/test_example.csv")
        st.write("Sample data: ", df_sample_data)
        st_pandas_to_csv_download_link(
            df_sample_data, file_name="test_example.csv")

    elif clf_name == 'Decision Tree':
        max_depth = st.sidebar.slider(
            'max_depth', 2, 30, help='Độ sâu của cây')
        params['max_depth'] = max_depth
    else:
        max_depth = st.sidebar.slider(
            'max_depth', 2, 30, help='Độ sâu của cây')
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider(
            'n_estimators', 1, 100, help='Số lượng cây')
        params['n_estimators'] = n_estimators
        min_samples_leaf = st.sidebar.slider(
            'min_samples_leaf', 1, 20, help='Số lượng mẫu tối thiểu cần thiết cho một nút lá')
        params['min_samples_leaf'] = min_samples_leaf
    return params


params = add_parameter(classifier_name)

uploaded_file = st.file_uploader(
    "Hãy nhập vào 1 file csv giống định dạng của file sample data")

X = pd.DataFrame()

if uploaded_file is not None:
    datasets = pd.read_csv(uploaded_file, index_col=0)
    print('datasets')
    print(datasets)
    X = datasets


def get_pred(clf_name):
    preds = None
    acc = 0
    label_preds = None
    predicted = False
    if clf_name == '':
        st.write("Xin mời lựa chọn mô hình phân lớp để đưa ra kết quả dự đoán")
        predicted = False

    elif clf_name == 'Decision Tree':
        decision_tree_with_max_depth = DecisionTreeClassifier(
            **params, random_state=42)
        decision_tree_with_max_depth.fit(X_train, y_train)
        print('line 94')
        print(X)
        preds = decision_tree_with_max_depth.predict_proba(X)[:, 1]
        print('preds')
        print(preds)

        y_val_preds = decision_tree_with_max_depth.predict_proba(X_val)[:, 1]
        print(y_val)
        acc = c_index(y_val, y_val_preds)

        label_preds = decision_tree_with_max_depth.predict(X)
        predicted = True
    else:
        random_forest_hypertuning = RandomForestClassifier(
            **params, random_state=42)
        random_forest_hypertuning.fit(X_train, y_train)
        preds = random_forest_hypertuning.predict_proba(X)[:, 1]

        y_val_preds = random_forest_hypertuning.predict_proba(X_val)[:, 1]
        acc = c_index(y_val, y_val_preds)

        label_preds = random_forest_hypertuning.predict(X)
        predicted = True

    return preds, acc, label_preds, predicted


if X.empty == True:
    st.write("Xin mời bạn nhập vào file dữ liệu dạng csv!")
else:
    preds, acc, label_preds, predicted = get_pred(classifier_name)
    X_risk = X.copy(deep=True)
    if predicted is False:
        st.write('Dữ liệu dùng để đưa ra dự đoán :', X_risk)
    else:
        st.write('Dữ liệu dùng để đưa ra dự đoán :', X_risk)
        results = np.array([])
        for i in range(label_preds.size):
            if label_preds[i] == True:
                results = np.append(results, 'Đột quỵ', axis=None)
            else:
                results = np.append(results, 'Không Đột quỵ', axis=None)
        y_risk = pd.DataFrame()
        y_risk.loc[:, 'Kết quả dự đoán'] = results
        y_risk.loc[:, 'Xác suất Đột quỵ'] = preds
        st.write(f'Mô hình đã lựa chọn để phân lớp : {classifier_name}')
        st.write(f'Kết quả dự đoán :', y_risk)
        st.sidebar.write(f'Độ chính xác: {round(acc*100, 2)}%')
