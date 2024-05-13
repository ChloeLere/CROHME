from sklearn import svm 

class Svc:
    def __init__(self, num_class=84):
        self.num_class = num_class
        self.model = self.build_model()
    
    def get_model(self):
        return self.model
    
    def build_model(self):
        clf = svm.SVC()
        return clf

    def compile_model(self, X_train_flat, y_train):
        self.model.fit(X_train_flat, y_train)
    
    def predict(self, X_test_flat):
        y_test_pred = self.model.predict(X_test_flat)
        return y_test_pred