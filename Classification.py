from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

mnist

X,y = mnist['data'],mnist["target"]

X.shape


y.shape

%matplotlib inline
import matplotlib
from matplotlib import pyplot as plt
some_digit = X[0]

%matplotlib inline
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()

i = list(y).index('5')

print(i)

y[0]

##Setting training and Testing set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

np.where(y_train=='5')

##shuffling the training index
shuffle_index = np.random.permutation(60000)
X_train,y_train = X[shuffle_index],y[shuffle_index]

##Training a binary classifier with y=5 or not 5
y_train_5 = (y_train=='5')
y_test_5  = (y_test=='5')


np.where(y_train_5==True)

#We are using an SGD classifier 
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)

sgd_clf.predict([some_digit])

##Model Performance
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring='accuracy')

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)



for train_index,test_index in skfolds.split(X_train,y_train_5):
    clone_sgd = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    X_test_folds = X_train[test_index]
    y_train_folds = y_train_5[train_index]
    y_test_folds = y_train_5[test_index]
    clone_sgd.fit(X_train_folds,y_train_folds)
    y_pred = clone_sgd.predict(X_test_folds)
    n_correct = sum(y_pred==y_test_folds)
    print(n_correct/len(y_pred))
    
