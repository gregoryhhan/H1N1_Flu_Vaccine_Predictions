import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from helper_functions import model_evaluation
from helper_functions import preprocessing_function

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
import pickle
from imblearn.over_sampling import SMOTE

import time
import sys
from multiprocessing import Process
from helper_functions import preprocessing_function

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Vaccine Classification Model')

st.sidebar.subheader('Action')
option_name = st.sidebar.selectbox('Options', ('Test our model', 'Simple Modelling', 'Comprehensive Modelling'))

if option_name == 'Simple Modelling':

    st.sidebar.subheader('Choose Target')
    target_name = st.sidebar.selectbox('Target', ('Seasonal Vaccines', 'H1N1 Vaccines'))

    st.sidebar.subheader('Choose Classifier')
    classifier_name = st.sidebar.selectbox(('Classifier'), ('Logistic Regression','Random Forest', 'Gradient Boost', 
                                                            'Histogram Gradient Boosting'))

    st.cache()
    def get_dataset(target_name):
        if target_name == 'Seasonal Vaccines':
            X_train = pd.read_csv('new/X_train_seasonal.csv')
            X_test = pd.read_csv('new/X_test_seasonal.csv')
            y_train = pd.read_csv('new/y_train_seasonal.csv')
            y_test = pd.read_csv('new/y_test_seasonal.csv')
        if target_name == 'H1N1 Vaccines':
            X_train = pd.read_csv('new/X_train_h1n1.csv')
            X_test = pd.read_csv('new/X_test_h1n1.csv')
            y_train = pd.read_csv('new/y_train_h1n1.csv')
            y_test = pd.read_csv('new/y_test_h1n1.csv')
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = get_dataset(target_name)

    def add_parameter_ui(classifier_name):
        params = dict()
        st.subheader('Model Hyperparameters')
        col1, col2 = st.columns(2)
        if classifier_name == 'Logistic Regression':
            with col1:
                C = st.selectbox('C value', (0.001, 0.01, 0.1, 1.0, 10, 100, 1000))
            with col2:
                penalty = st.selectbox('Penalty', ('l1', 'l2', 'elasticnet'))
            params['C'] = C
            params['penalty'] = penalty
        if classifier_name == 'Random Forest':
            with col1:
                max_depth = st.slider('Max Depth', 1, 15)
            with col2:
                max_features = st.slider('Max Features', 0.1, 1.0)
            params['max_depth'] = max_depth
            params['max_features'] = max_features
        if classifier_name == 'Gradient Boost':
            with col1:
                max_depth = st.slider('Max Depth', 1, 15)
            with col2:
                learning_rate = st.slider('Learning Rate', 0.1, 1.0)
            params['max_depth'] = max_depth
            params['learning_rate'] = learning_rate
        if classifier_name == 'Histogram Gradient Boosting':
            with col1:
                max_depth = st.slider('Max Depth', 1, 15)
            with col2:
                learning_rate = st.slider('Learning Rate', 0.1, 1.0)
            params['max_depth'] = max_depth
            params['learning_rate'] = learning_rate
        return params

    params = add_parameter_ui(classifier_name)

    make_model = st.sidebar.button('Make model')

    st.cache()
    def get_classifier(classifier_name, params):
        if classifier_name == 'Logistic Regression':
            clf = LogisticRegression(fit_intercept=False, 
                                    max_iter = 300,
                                    solver = 'saga',
                                    penalty = params['penalty'],
                                    C = params['C'])

        if classifier_name == 'Random Forest':
            clf = RandomForestClassifier(
                criterion = 'entropy',
                n_estimators = 170,
                min_samples_leaf = 190,
                max_features = params['max_features'],
                max_depth = params['max_depth']
            )

        if classifier_name == 'Gradient Boost':
            clf = GradientBoostingClassifier(n_estimators = 150,
                                            learning_rate = params['learning_rate'],
                                            max_depth = params['max_depth'],
                                            min_samples_leaf = 190,
                                            min_samples_split = 500)

        if classifier_name == 'Histogram Gradient Boosting':   
            clf = HistGradientBoostingClassifier(
                                                learning_rate = params['learning_rate'],
                                                max_depth = params['max_depth'],
                                                min_samples_leaf = 190
                                                )
        return clf

    col1, col2, col3 = st.columns(3)
    with col1:
        graph_confusion_matrix = st.checkbox('Confusion Matrix')
    with col2:
        graph_ROC_Curve = st.checkbox('ROC Curve') 
    with col3:
        graph_precision_recall_curve = st.checkbox('Precision Recall Curve')

    if make_model:

        clf = get_classifier(classifier_name, params)

        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        auc = roc_auc_score(y_test, y_proba[:, 1])
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        col1, col2, col3 = st.columns(3)

        st.subheader(f'{classifier_name} results')

        #with col1:
            #st.write(###
            #test)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f'AUC: {round(auc, 3)}')
        with col2:    
            st.write(f"Accuracy: {round(acc, 3)}")
        with col3:    
            st.write(f"Recall: {round(recall, 3)}")
        with col4:    
            st.write(f'Precision: {round(precision, 3)}')

        if graph_confusion_matrix:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(clf, X_test, y_test, cmap = plt.cm.Blues)
            st.pyplot()
        
        if graph_ROC_Curve:
            st.subheader('ROC Curve')
            plot_roc_curve(clf, X_test, y_test)
            st.pyplot()

        if graph_precision_recall_curve:
            st.subheader('Precision Recall Curve')
            plot_precision_recall_curve(clf, X_test, y_test)
            st.pyplot()

if option_name == 'Test our model':
    col1, col2 = st.columns(2)

    with col1:
        vaccine_type = st.selectbox('Vaccine Type', ('H1N1', 'Seasonal'))
    with col2:
        model_type = st.selectbox('Model Type', ('Random Forest', 'Naive Bayes'))
    ''
    with st.form(key = 'form1'):
        col1, col2 = st.columns(2)
        with col1:
            opinion_vacc_effective = st.selectbox('How do you feel about the effectiveness of the vaccine?',
                                                        ('Not at all effective',
                                                        'Not very effective',
                                                        'Unsure',
                                                        'Somewhat effective',
                                                        'Very effective'))
            opinion_vacc_effective_num = {'Not at all effective': 1, 
                                        'Not very effective': 2,
                                        'Unsure': 3,
                                        'Somewhat effective': 4,
                                        'Very effective': 5}[opinion_vacc_effective]
            ''
            concern = st.selectbox('How concerned are you about the flu?', 
                                    ('Not at all concerned',
                                    'Not very concerned',
                                    'Somewhat concerned',
                                    'Very concerned')
                                    )
            concern_num = {'Not at all concerned': 0,
                            'Not very concerned': 1,
                            'Somewhat concerned': 2,
                            'Very concerned': 3}[concern]       
            ''
            concern_sick_from_vacc = st.selectbox('Are you worried abou being sick from the vaccine?', 
                                                    ('Not at all worried',
                                                    'Not very worried',
                                                    'Unsure',
                                                    'Somewhat worried',
                                                    'Very worried')
                                                    )
            concern_sick_from_vacc_num = {'Not at all worried': 1,
                                            'Not very worried': 2,
                                            'Unsure': 3,
                                            'Somewhat worried': 4,
                                            'Very worried': 5}[concern_sick_from_vacc]                                    

        with col2:
            doctor_recc = st.radio('Were you recommended by a doctor?', ('Yes', 'No'))
            doctor_recc_num = {'Yes': 1, 'No': 0}[doctor_recc]
            ''
            health_worker = st.radio('Are you a health worker?', ('Yes', 'No'))
            health_worker_num = {'Yes': 1, 'No': 0}[health_worker]
            ''
            opinion_risk = st.slider('How much of a risk is getting the flu without being vaccinated to you?', 1, 5)

        make_prediction = st.form_submit_button(label = 'Make Prediction')
    
    if make_prediction:

        if vaccine_type == 'H1N1':
            input_dict = {'doctor_recc_h1n1': doctor_recc_num,
                        'opinion_h1n1_risk': opinion_risk,
                        'opinion_h1n1_vacc_effective': opinion_vacc_effective_num,
                        'h1n1_concern': concern_num,
                        'opinion_h1n1_sick_from_vacc': concern_sick_from_vacc_num,
                        'health_worker': health_worker_num}

            input_frame = pd.DataFrame.from_dict(input_dict, orient='index').T

            if model_type == 'Random Forest':
                    model = pickle.load(open('models/h1n1_rf_model.pickl', 'rb'))
            if model_type == 'Naive Bayes':
                model = pickle.load(open('models/h1n1_naive_bayes_model.pickl', 'rb'))

            prediction = model.predict(input_frame)[0]

            if prediction == 1:
                st.write('This individual was predicted to have taken the H1N1 vaccine.')
            else:
                st.write('This person was predicted not to have taken the H1N1 vaccine.')

        if vaccine_type == 'Seasonal':
            input_dict = {'doctor_recc_seasonal': doctor_recc_num,
                        'opinion_seas_risk': opinion_risk,
                        'opinion_seas_vacc_effective': opinion_vacc_effective_num,
                        'h1n1_concern': concern_num,
                        'opinion_seas_sick_from_vacc': concern_sick_from_vacc_num,
                        'health_worker': health_worker_num}

            input_frame = pd.DataFrame.from_dict(input_dict, orient='index').T

            if model_type == 'Random Forest':
                    model = pickle.load(open('models/seasonal_rf_model.pickl', 'rb'))
            if model_type == 'Naive Bayes':
                model = pickle.load(open('models/seasonal_naive_bayes_model.pickl', 'rb'))

            prediction = model.predict(input_frame)[0]

            if prediction == 1:
                st.write('This individual was predicted to have taken the seasonal flu vaccine.')
            else:
                st.write('This person was predicted not to have taken the seasonal flu vaccine.')

if option_name == 'Comprehensive Modelling':
    
    st.sidebar.subheader('Choose Target')
    target_name = st.sidebar.selectbox('Target', ('Seasonal Vaccines', 'H1N1 Vaccines'))

    st.sidebar.subheader('Choose Classifier')
    classifier_name = st.sidebar.selectbox(('Classifier'), ('Logistic Regression','Random Forest', 'Gradient Boost', 
                                                            'Histogram Gradient Boosting'))

    X = pd.read_csv('data/training_set_features.csv').drop('respondent_id', axis = 1)
    y = pd.read_csv('data/training_set_labels.csv')

    column_options = st.multiselect(
                    'Select the features to exclude',
                    X.columns
                    )

    def add_parameter_ui(classifier_name):
        params = dict()
        st.subheader('Model Hyperparameters')
        col1, col2 = st.columns(2)
        if classifier_name == 'Logistic Regression':
            with col1:
                solver = st.selectbox('Solver',('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
            with col2:
                penalty = st.selectbox('Penalty', ('l1', 'l2', 'elasticnet'))
            C = st.slider('C value', 0.001, 1000.0)
            params['solver'] = solver
            params['C'] = C
            params['penalty'] = penalty
        if classifier_name == 'Random Forest':
            with col1:
                criterion = st.selectbox('Criterion', ('entropy', 'gini'))
                max_depth = st.slider('Max Depth', 1, 15)
                n_estimators = st.slider('n_estimators', 1, 1000)
            with col2:
                max_features = st.selectbox('Max Features', ('sqrt', 'log2'))
                min_samples_leaf = st.slider('min_samples_leaf', 1, 200)
                max_iter = st.slider('max_iter', 1, 200)
            params['criterion'] = criterion
            params['min_samples_leaf'] = min_samples_leaf
            params['n_estimators'] = n_estimators
            params['max_depth'] = max_depth
            params['max_features'] = max_features
        if classifier_name == 'Gradient Boost':
            with col1:
                max_depth = st.slider('Max Depth', 1, 15)
            with col2:
                learning_rate = st.slider('Learning Rate', 0.1, 1.0)
            params['max_depth'] = max_depth
            params['learning_rate'] = learning_rate
        if classifier_name == 'Histogram Gradient Boosting':
            with col1:
                max_depth = st.slider('Max Depth', 1, 15)
            with col2:
                learning_rate = st.slider('Learning Rate', 0.1, 1.0)
            params['max_depth'] = max_depth
            params['learning_rate'] = learning_rate
        return params

    params = add_parameter_ui(classifier_name)

    st.cache()
    def get_classifier(classifier_name, params):
        if classifier_name == 'Logistic Regression':
            clf = LogisticRegression(fit_intercept=False,
                                    solver = params['solver'],
                                    penalty = params['penalty'],
                                    C = params['C'])

        if classifier_name == 'Random Forest':
            clf = RandomForestClassifier(criterion = params['criterion'],
                                        n_estimators = params['n_estimators'],
                                        min_samples_leaf = params['min_samples_leaf'],
                                        max_features = params['max_features'],
                                        max_depth = params['max_depth'])

        if classifier_name == 'Gradient Boost':
            clf = GradientBoostingClassifier(n_estimators = 150,
                                            learning_rate = params['learning_rate'],
                                            max_depth = params['max_depth'],
                                            min_samples_leaf = 190,
                                            min_samples_split = 500)

        if classifier_name == 'Histogram Gradient Boosting':   
            clf = HistGradientBoostingClassifier(
                                                learning_rate = params['learning_rate'],
                                                max_depth = params['max_depth'],
                                                min_samples_leaf = 190
                                                )
        return clf

    columns_to_exclude = [i for i in column_options]

    comprehensive_model_button = st.sidebar.button('Make model')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        graph_confusion_matrix = st.checkbox('Confusion Matrix')
    with col2:
        graph_ROC_Curve = st.checkbox('ROC Curve') 
    with col3:
        graph_precision_recall_curve = st.checkbox('Precision Recall Curve')

    if comprehensive_model_button:
        X = X.drop(columns_to_exclude, axis = 1)

        if  target_name == 'Seasonal Vaccines':
            y = y['seasonal_vaccine']
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            process = preprocessing_function(X_train)
            X_train_processed = process.fit_transform(X_train)
            X_test_processed = process.transform(X_test)
            
            X_train = X_train_processed
            X_test = X_test_processed


        if  target_name == 'H1N1 Vaccines':
            y = y['h1n1_vaccine']
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            
            process = preprocessing_function(X_train)
            X_train_processed = process.fit_transform(X_train)
            X_test_processed = process.transform(X_test)

            sm = SMOTE()
            X_train_smote, y_train_smote = sm.fit_resample(X_train_processed, y_train_processed)
            X_train = X_train_smote
            y_train = y_train
    
        clf = get_classifier(classifier_name, params)

        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        auc = roc_auc_score(y_test, y_proba[:, 1])
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        col1, col2, col3 = st.columns(3)

        st.subheader(f'{classifier_name} results')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f'AUC: {round(auc, 3)}')
        with col2:    
            st.write(f"Accuracy: {round(acc, 3)}")
        with col3:    
            st.write(f"Recall: {round(recall, 3)}")
        with col4:    
            st.write(f'Precision: {round(precision, 3)}')

        if graph_confusion_matrix:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(clf, X_test, y_test, cmap = plt.cm.Blues)
            st.pyplot()
        
        if graph_ROC_Curve:
            st.subheader('ROC Curve')
            plot_roc_curve(clf, X_test, y_test)
            st.pyplot()

        if graph_precision_recall_curve:
            st.subheader('Precision Recall Curve')
            plot_precision_recall_curve(clf, X_test, y_test)
            st.pyplot()
            






    
    


    



    
    
   