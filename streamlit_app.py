#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    
    st.title('Compare Linear Regression and Naive Bayes Classifiers')
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
 
    st.write('Logistic Regression:')
    text = """Strengths: \nMore flexible: Can capture complex relationships between 
    features and classes, even when they are non-linear. No strong independence assumption: 
    Doesn't rely on the assumption that features are independent, which can be 
    helpful for overlapping clusters."""
    st.write(text)
    text = """Weaknesses: \nOverfitting potential: Can overfit the training data when 
    dealing with high dimensionality 
    or small datasets."""
    st.write(text)

    st.write('Naive Bayes')
    text = """Strengths: \nEfficient: Works well with high-dimensional datasets 
    due to its simplicity. 
    Fast training: Requires less training time compared to logistic regression. 
    Interpretable: Easy to understand the contribution of each feature to the prediction."""
    st.write(text)

    text = """Weaknesses:\nIndependence assumption: Relies on the strong 
    assumption of feature independence, which can be violated in overlapping clusters, 
    leading to inaccurate predictions."""
    st.write(text)

    # Create the logistic regression 
    clf = GaussianNB() 
    options = ['Naive Bayes', 'Logistic Regression']
    selected_option = st.selectbox('Select the classifier', options)
    if selected_option=='Logistic Regression':
        clf = LogisticRegression(C=1.0, class_weight=None, 
            dual=False, fit_intercept=True,
            intercept_scaling=1, max_iter=100, multi_class='auto',
            n_jobs=1, penalty='l2', random_state=42, solver='lbfgs',
            tol=0.0001, verbose=0, warm_start=False)
    else:
        clf = GaussianNB()
        
    if st.button('Start'):
        
        df = pd.read_csv('binary_data.csv', header=0)
        
        st.subheader('The Dataset')
        # display the dataset
        st.dataframe(df, use_container_width=True)  

        #load the data and the labels
        X = df.values[:,0:-1]
        y = df.values[:,-1]          
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=0.2, random_state=42)
        
        clf.fit(X_train,y_train)
        y_test_pred = clf.predict(X_test)
        st.subheader('Confusion Matrix')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))
        st.subheader('VIsualization')
        visualize_classifier(clf, X_test, y_test_pred)

def visualize_classifier(classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)
    
    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Specify the title
    ax.set_title(title)
    
    # Choose a color scheme for the plot
    ax.pcolormesh(x_vals, y_vals, output, cmap="OrRd")
    
    # Overlay the training points on the plot
    ax.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap="gist_rainbow")
    
    # Specify the boundaries of the plot
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())
    
    # Specify the ticks on the X and Y axes
    ax.set_xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    ax.set_yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    
    st.pyplot(fig)
    
#run the app
if __name__ == "__main__":
    app()
