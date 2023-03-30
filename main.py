import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.svm
import sklearn.ensemble
import matplotlib.pyplot as plt

log_df = pd.read_csv("data/log_mini.csv")
log_df.rename(columns={"track_id_clean": "track_id"}, inplace=True)
tf_df = pd.read_csv("data/tf_mini.csv")
df = log_df.merge(tf_df, on=["track_id"], how="left")

train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
x_columns = tf_df.columns[
    ~tf_df.columns.to_series().isin(["duration", "release_year", "track_id", "key", "mode"])]
y_columns = log_df.columns[log_df.columns.str.contains("skip")]
label_map = y_columns.to_numpy()


def process_y(y_df):
    return np.argmax(y_df[y_columns].to_numpy().astype(int), axis=1)


train = train.head(100000)
test = test.head(10000)
train_X = train[x_columns]
train_y = process_y(train[y_columns])
test_X = test[x_columns]
test_y = process_y(test[y_columns])

model = sklearn.ensemble.RandomForestClassifier()
print("Fitting...")
model.fit(train_X, train_y)
print("Fit. Testing...")
preds = model.predict(test_X)
print("Results:")
print(sklearn.metrics.classification_report(test_y, preds))
print(label_map)

importances = model.feature_importances_
feature_names = x_columns
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
