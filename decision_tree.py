import pandas as pd
from sklearn import tree
import graphviz
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv("car.csv")


# data pre-processing
df_dummies = pd.get_dummies(df)
attribute_names = df_dummies.columns.tolist()
feature_names = attribute_names[:-2]
target_names = attribute_names[-2:]

X_train = df_dummies.iloc[:int(0.75 * len(df)),:-2]
y_train = df_dummies.iloc[:int(0.75 * len(df)), -2]
X_test = df_dummies.iloc[int(0.75 * len(df)):,:-2]
y_test = df_dummies.iloc[int(0.75 * len(df)):, -2].values.tolist()



def run(CRITERION, MAX_DEPTH, MIN_SPLIT):
    clf = tree.DecisionTreeClassifier(criterion=CRITERION, max_depth=MAX_DEPTH, min_samples_split=MIN_SPLIT)
    clf = clf.fit(X_train, y_train)


    # Define feature values for binary features
    feature_values = [['0', '1'] for _ in range(len(feature_names))]

    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names,
                                    class_names = target_names, filled=True, rounded=True,
                                    special_characters=True)

    # Change the output setting to let categorical variables display 0/1 rather than <= 0.5
    for i, feature_name in enumerate(feature_names):
        for j, feature_value in enumerate(feature_values[i]):
            dot_data = dot_data.replace(f'{feature_name} &le; {j + 0.5}', f'{feature_name} = {feature_value}')

    # print(dot_data)
    graph = graphviz.Source(dot_data)
    graph.render("Results/{}_{}_{}".format(CRITERION, MAX_DEPTH, MIN_SPLIT)) 
    
    cross_validation_lst = cross_val_score(clf, X_train, y_train, cv=10).tolist()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.title("ROC curve")
    plt.show()
    
    # cross_validation_lst.append(correct_num / len(X_test))

    return cross_validation_lst
    
criterions = ['gini', 'entropy', 'log_loss']
max_depth = [_ for _ in range(3, 8)] + [None]
min_samples = [_ for _ in range(5, 55, 5)]
scores = []
# for c in criterions:
#     for d in max_depth:
#         for s in min_samples:
#             result = run(c,d,s)
#             result.extend([c,d,s])
#             scores.append(result)

    
# result_df = pd.DataFrame(scores)
# result_df.to_csv("Running results.csv")

run('gini', 3, 10)
