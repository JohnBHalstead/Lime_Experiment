# %%
# Packages, libraries, data, etc.
import time
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Models
nn_clf = MLPClassifier(hidden_layer_sizes=(15,), alpha=0.1, max_iter=100000)
ada_clf = AdaBoostClassifier(learning_rate=1.25, n_estimators=175)
rf_clf = RandomForestClassifier(n_estimators=1666, max_features="auto", min_samples_split=2, min_samples_leaf=2,
                                max_depth=20, bootstrap=True, n_jobs=1)
knn_clf = KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="auto", leaf_size=40, p=1)

# load and organize Wisconsin Breast Cancer Data 
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Look at the data
print(label_names)
print(labels)
print(feature_names)
print(features.shape)  # (569, 30)

# Random split data
X_tng, X_val, y_tng, y_val = train_test_split(features, labels, test_size=0.33, random_state=42)

print(X_tng.shape) # (381, 30)
print(X_val.shape) # (188, 30)

# %%
# modeling (optimized by random search in shap.py)
start = time.time()
for clf in (knn_clf, rf_clf, nn_clf, ada_clf):
    clf.fit(X_tng, y_tng)
    y_hat = clf.predict(X_val)
    print(clf.__class__.__name__, roc_auc_score(y_val, y_hat))

end = time.time()
print(end - start)

# KNeighborsClassifier 0.950289872949303
# RandomForestClassifier 0.9577525595164673
# MLPClassifier 0.9544221043542618
# AdaBoostClassifier 0.9801406192179597
# Time 2.747337818145752 seconds

# %%
# Lime explainer for all classifiers
explain_this = lime.lime_tabular.LimeTabularExplainer(X_tng, mode="classification",training_labels=feature_names, class_names=label_names)

# asking for explanation of row i for LIME model
# Lime requires probability rather then binomial
# AdaBoost Classifier
i = 1
start = time.time()
explain_oneRow = explain_this.explain_instance(X_val[ i , : ], ada_clf.predict_proba, num_features=10 )
end = time.time()
print(end-start)
explain_oneRow.show_in_notebook(show_table=True)
explain_oneRow.as_pyplot_figure()
plt.savefig('AdaLime.png', bbox_inches="tight", pad_inches=1)

# Random Forest Classifier
i = 1
start = time.time()
explain_oneRow = explain_this.explain_instance(X_val[ i , : ], rf_clf.predict_proba, num_features=10 )
end = time.time()
print(end-start)
explain_oneRow.show_in_notebook(show_table=True)
explain_oneRow.as_pyplot_figure()
plt.savefig('RfLime.png', bbox_inches="tight", pad_inches=1)

# Neural Network Classifier
i = 1
start = time.time()
explain_oneRow = explain_this.explain_instance(X_val[ i , : ], nn_clf.predict_proba, num_features=10 )
end = time.time()
print(end-start)
explain_oneRow.show_in_notebook(show_table=True)
explain_oneRow.as_pyplot_figure()
plt.savefig('NnLime.png', bbox_inches="tight", pad_inches=1)

# KNN Classifier
i = 1
start = time.time()
explain_oneRow = explain_this.explain_instance(X_val[ i , : ], knn_clf.predict_proba, num_features=10 )
end = time.time()
print(end-start)
explain_oneRow.show_in_notebook(show_table=True)
explain_oneRow.as_pyplot_figure()
plt.savefig('KnnLime.png', bbox_inches="tight", pad_inches=1)

# %%
# looping the models' Lime explaination with plots for observations i
start = time.time()
explain_loop = lime.lime_tabular.LimeTabularExplainer(X_tng, mode="classification",training_labels=feature_names, class_names=label_names)
j = 1
for clf in (knn_clf, rf_clf, nn_clf, ada_clf):
    explain_oneRow = explain_loop.explain_instance(X_val[ i , : ],clf.predict_proba, num_features=10 )
    print(clf.__class__.__name__)
    explain_oneRow.as_pyplot_figure()

end = time.time()
print("total time", end - start)
