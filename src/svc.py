from sklearn import svm 

class Svc:
    def __init__(self, num_class=84):
        self.num_class = num_class
    
    def build_model(self):
        clf = svm.SVC()
        return clf

    def compile_model(self, clf, X_train_flat, y_train):
        clf.fit(X_train_flat, y_train)
        return clf
