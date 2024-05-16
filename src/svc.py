from sklearn import svm 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay

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
    
    def display_history_v1(self):
        results = self.grid_search.cv_results_
        print(results.keys())

        means_test = results['mean_test_score']
        stds_test = results['std_test_score']

        ## Getting indexes of values per hyper-parameter
        masks=[]
        masks_names= list(self.grid_search.best_params_.keys())
        for p_k, p_v in self.grid_search.best_params_.items():
            masks.append(list(results['param_'+p_k].data==p_v))

        params=self.grid_search.param_grid

        ## Ploting results
        fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
        fig.suptitle('Score per parameter')
        fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
        pram_preformace_in_best = {}
        for i, p in enumerate(masks_names):
            m = np.stack(masks[:i] + masks[i+1:])
            pram_preformace_in_best
            best_parms_mask = m.all(axis=0)
            best_index = np.where(best_parms_mask)[0]
            x = np.array(params[p])
            y_1 = np.array(means_test[best_index])
            e_1 = np.array(stds_test[best_index])
            ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
            ax[i].set_xlabel(p.upper())

        plt.legend()
        plt.savefig("grid_search_results_svc.png")

        plt.show()

    def display_history_v2(self):
        results = self.grid_search.cv_results_
        mean_test_score = results['mean_test_score']
        margin = 0.000001
        y_min = np.min(mean_test_score) - margin * np.ptp(mean_test_score)
        y_max = np.max(mean_test_score) + margin * np.ptp(mean_test_score)

        plt.figure(figsize=(12, 6))
        
        plt.plot(mean_test_score, marker='o', linestyle='-', label='Mean Test Score')
        plt.title('Evolution of the mean test score')
        plt.xlabel('epoch')
        plt.ylabel('Mean Test Score')
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()

        plt.savefig("grid_search_results_svc.png")
        
        plt.show()


    def plot_confusion_matrics(self, y_test, y_test_pred):
        cm = confusion_matrix(y_test, y_test_pred, self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        plt.savefig("confusion_matrics_svc.png")
        plt.show()

    def plot_roc_curve(self, y_test, y_test_pred):
        fpr, tpr, _ = roc_curve(y_test, y_test_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve_svc.png")
        plt.show()

    def plot_sensitivity_specificity(self, y_test, y_test_pred):
        specificity, sensitivity, _ = precision_recall_curve(y_test, y_test_pred)
        plt.step(sensitivity, specificity, color='b', alpha=0.2, where='post')
        plt.fill_between(sensitivity, specificity, step='post', alpha=0.2, color='b')
        plt.xlabel('Sensitivity')
        plt.ylabel('Specificity')
        plt.title('Specificity-sensitivity Curve')
        plt.savefig("sensitivity_specificity_svc.png")
        plt.show()

