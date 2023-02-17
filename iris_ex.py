# 뭔신 ? https://www.youtube.com/watch?v=sqorJR4TNmg&list=PLsGh7Wc318kgZlJQ1Rhb9lUDTshc9-MzW&index=3
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
# 데이터를 불러와 X,Y 로 나누고 RandomForestClassifier 객체에서 fit, predict

iris = datasets.load_iris()
st.write("""
# Iris 예측 웹
""")

st.sidebar.header("입력값")

def user_input_features():
    sepal_length = st.sidebar.slider('꽃받침(Sepal)길이', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('꽃받침(Sepal)넓이', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('꽃잎(petal) length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('꽃잎(petal) width', 0.1, 2.5, 0.2)
    data = {
        'sepal_length' : sepal_length,
        'sepal_width' : sepal_width,
        'petal_length' : petal_length,
        'petal_width' : petal_width
    }
    freautres = pd.DataFrame(data, index=[0])
    return freautres

df = user_input_features()

st.subheader('사용자 입력 파라미터')
st.write(df)

X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('클레스 레이블 및 해당 색인 번호')
st.write(iris.target_names)

st.subheader('예측')
st.write(iris.target_names[prediction])

st.subheader('예측 확률')
st.write(prediction_proba)


