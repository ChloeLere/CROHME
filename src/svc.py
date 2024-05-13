from sklearn import svm 
from sklearn.model_selection import GridSearchCV

class Svc:
    def __init__(self, num_class=84):
        self.num_class = num_class
        self.model = self.build_model()
        self.grid_search = None
    
    def get_model(self):
        return self.model
    
    def build_model(self):
        clf = svm.SVC()
        return clf

    def compile_model(self, X_train_flat, y_train, param_grid):
        grid_search = GridSearchCV(self.model, param_grid)
        grid_search.fit(X_train_flat, y_train)
        self.model = grid_search.best_estimator_
        self.grid_search = grid_search


    def display_grid_search_info(self):
        if self.grid_search == None:
            return
        print("Best parameters found:", self.grid_search.best_params_)
        print("Grid search results:", self.grid_search.cv_results_)
        print("Best score:", self.grid_search.best_score_)
        print("Best estimator:", self.grid_search.best_estimator_)
    
    
    def predict(self, X_test_flat):
        y_test_pred = self.model.predict(X_test_flat)
        return y_test_pred