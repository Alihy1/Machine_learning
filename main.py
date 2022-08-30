import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

test_set = pd.read_csv('archive-3/Training.csv')
test_set.drop('Unnamed: 133', axis=1, inplace=True)

x = test_set.drop('prognosis', axis=1)
y = test_set['prognosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

predict = tree.predict(x_test)
acc = tree.score(x_test, y_test)
print('Accuracy: {:.2f}%'.format(acc*100))

fi = pd.DataFrame(tree.feature_importances_*100, x_test.columns, columns=['importance'])
fi.sort_values(by='importance', ascending=False, inplace=True)
print(fi)


