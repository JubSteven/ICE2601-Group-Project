from decision_tree_modified import ID3DecisionTree
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt

def cross_validation(X, y, decision_tree, k=2):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        decision_tree.fit(X_train, y_train)
        y_pred = decision_tree.predict(X_test)
        correct_predictions = sum([1 for i in range(len(y_test)) if y_pred[i] == y_test[i]])
        accuracy = correct_predictions / len(y_test)
        accuracies.append(accuracy)
    avg_accuracy = sum(accuracies) / k
    return accuracies, avg_accuracy

df = pd.read_csv("car.csv")

# data pre-processing
df_dummies_train = pd.get_dummies(df)



X = df_dummies_train.iloc[:len(df),:-2].values.tolist()
y = df_dummies_train.iloc[:len(df), -2].values.tolist()
X_train = df_dummies_train.iloc[:int(0.75 * len(df)),:-2].values.tolist()
y_train = df_dummies_train.iloc[:int(0.75 * len(df)), -2].values.tolist()
X_test = df_dummies_train.iloc[int(0.75 * len(df)):,:-2].values.tolist()
y_test = df_dummies_train.iloc[int(0.75 * len(df)):, -2].values.tolist()

print(df_dummies_train.iloc[0,:-2])

result_df = {"acc": [], "p": [], "q": []}

for p in range(0, 30, 5):
    for q in range(1, 11, 1):
        clf = ID3DecisionTree(p=p/100, q=q/10)        
        clf.fit(X_train, y_train)

        pred = clf.predict(X_train)
        count = 0
        # print(pred)
        for i in range(len(pred)):
            if pred[i] == y_train[i]:
                count += 1
                
        print("[TRAINING ACCURACY] {} p {} q {}".format(count / len(pred), p/100, q/10))
        result_df["acc"].append(count / len(pred))
        result_df["p"].append(p/100)
        result_df["q"].append(q/10)
        
# sns.set_theme(style="whitegrid")
sns.scatterplot(data=result_df, x="p", y="acc", hue="q", palette="deep", s=50)
plt.title("The effect of p and q on accuracy (training)")
plt.show()


