import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
iris = load_iris()

# Train the k-NN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iris.data, iris.target)

# Streamlit app
st.title('Iris Flower Prediction')
st.write('Enter the measurements of the iris flower to predict its species.')

# User inputs
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, step=0.1)

# Prediction
if st.button('Predict'):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = knn.predict(input_data)
    species = iris.target_names[prediction][0]
    st.write(f'The predicted species is: {species}')