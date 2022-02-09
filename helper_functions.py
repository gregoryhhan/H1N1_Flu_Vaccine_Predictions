def model_evaluation(X_train, X_test, y_train, y_test,
                         baseline_models, 
                         preprocessor,
                         folder_name = None, 
                         ):
    
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import CategoricalNB

    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier

    from sklearn.svm import SVC
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.model_selection import RandomizedSearchCV
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import plot_roc_curve
    from sklearn.metrics import roc_curve, auc
    
    
    # Create a summary dictionary
    summary_dict = {}
    
    for name, model in baseline_models.items():
        
        # transform the features    
        processor = model['preprocessor']
        X_train_processed = processor.fit_transform(X_train)
        X_test_processed = processor.transform(X_test)
    
        # Cross validation
        model['train_accuracy_score'] = np.mean(cross_val_score(model['regressor'], 
                                                        X_train_processed, y_train.values.ravel(), 
                                                        scoring="accuracy", cv=5))
    
        train_accuracy_score = model['train_accuracy_score']
    
        # fit the new model and make predictions
        new_model = model['regressor']
        new_model.fit(X_train_processed, y_train.values.ravel())
        preds = new_model.predict(X_test_processed)
        y_score = new_model.predict_proba(X_test_processed)

        # get our scoring metrics
        model['test_accuracy_score'] = accuracy_score(y_test, preds)
        test_accuracy_score = model['test_accuracy_score']
        
        model['auc_score'] = roc_auc_score(y_test, y_score[:,1])
        auc_score = model['auc_score']
        
        model['recall_score'] = recall_score(y_test, preds)
        model['precision_score'] = f1_score(y_test, preds)
        model['f1_score'] = precision_score(y_test, preds)
        
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        
        # Visualisations
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1])
        model['fpr'] = fpr
        model['tpr'] = tpr
        model['thresholds'] = thresholds
    
        # Saving the model
        if folder_name == None:
            pass
        else:
            os.makedirs(f'models/{name}/{folder_name}') 
            filepath = f'models/{name}/{folder_name}/baseline_model.pickl'
            pickle.dump(new_model, open(filepath, 'wb'))
        
        #Place everything into a dictionary and place that into the summary list
        summary_dict.update({name: {
                                   'train_score': train_accuracy_score, 'test_score': test_accuracy_score,
                                   'recall': recall, 'precision': precision, 'f1': f1,
                                   'auc': auc_score, 'tpr': tpr, 'fpr': fpr
                                   }})

    return summary_dict
    
def preprocessing_function(X_train):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    X_train = X_train

    numericals = []
    non_numericals = []

    for column in X_train.columns:
        if X_train[column].dtype == 'float64':
            numericals.append(column)
        if X_train[column].dtype == 'object':
            non_numericals.append(column)

    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator = True)),
                               ('scaler', StandardScaler())])

    categorical_transformer = Pipeline([('cat_imputer', SimpleImputer(strategy='most_frequent', add_indicator = True)),
                                    ('encoder', OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numericals),
            ("cat", categorical_transformer, non_numericals),
        ]
    )

    return preprocessor