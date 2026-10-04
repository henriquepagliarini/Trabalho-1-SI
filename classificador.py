import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import joblib

class Classificador:
    def __init__(self, dataset, model, parameters, n_folds):
        self.dataset = dataset
        self.model = model
        self.parameters = parameters
        self.n_folds = n_folds

        self.results = {}
        self.best_result = None
        self.best_model = None
        
    def train(self):
        self.clf = GridSearchCV(
            self.model,
            self.parameters,
            cv=self.n_folds,
            scoring='f1_macro',
            return_train_score=True
        )

        self.clf.fit(self.dataset.Xm, self.dataset.Ym)

        self.results = self.clf.cv_results_
    
    def retrain(self):
        print(f"\n========= Retreino - {self.__class__.__name__.upper()} =========")
        
        self.best_model = self.model.set_params(**self.best_result['params'])
        self.best_model.fit(self.dataset.Xm, self.dataset.Ym)
        
        y_pred_retrain = self.best_model.predict(self.dataset.Xm)
        acc_train = accuracy_score(self.dataset.Ym, y_pred_retrain) * 100
        print(f"Acuracia com dados de treino: {acc_train:.2f}%")
        
    def test(self, test_dataset):
        print(f"\n========= Teste - {self.__class__.__name__.upper()} =========")

        y_pred_test = self.best_model.predict(test_dataset.Xm)
        Ym = test_dataset.Ym
        
        precision = precision_score(Ym, y_pred_test, average='macro')
        recall = recall_score(Ym, y_pred_test, average='macro')
        f1 = f1_score(Ym, y_pred_test, average='macro')
        accuracy = accuracy_score(Ym, y_pred_test) * 100
        
        print(f"Precisão (macro): ...........{precision:.4f}")
        print(f"Recall (macro): .............{recall:.4f}")
        print(f"F1 Score (macro): ...........{f1:.4f}")
        print(f"Acuracia com dados de teste: {accuracy:.2f}%")
        
        ConfusionMatrixDisplay.from_predictions(Ym, y_pred_test, display_labels=['G', 'Y', 'R', 'B'])
        plt.gcf().canvas.manager.set_window_title(f"Matriz de confusão - {self.__class__.__name__.upper()}")
    
    def save_model(self):
        filename = f"melhor_{self.__class__.__name__.lower()}.joblib"
        joblib.dump(self.best_model, filename)
        print(f"Modelo salvo em: {filename}")
    
    def process_results(self):
        res = self.results
        self.models = []
        
        for i, params in enumerate(res['params']):
            
            train_scores = np.array([
                res[f'split{f}_train_score'][i]
                for f in range(self.n_folds)
                ])
            val_scores = np.array([
                res[f'split{f}_test_score'][i]
                for f in range(self.n_folds)
                ])

            dif = (np.abs(train_scores - val_scores))
            
            model = {
                'params': params,

                'train_scores': train_scores,
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),

                'val_scores': val_scores,
                'val_mean': np.mean(val_scores),
                'val_std': np.std(val_scores),

                'differences': dif,
                'dif_mean': np.mean(dif),
                'dif_std': np.std(dif)
            }
            
            self.models.append(model)
    
    def show_results(self):
        print(f"\n=============== Classificador {self.__class__.__name__.upper()} ===============")
        for i, model in enumerate(self.models):

            print(f"\nModelo {i + 1}")
            print(f"Parâmetros: {model['params']}")

            print(
                f"Treino:    F1 por fold: "
                f"{[f'{v:.5f}' for v in model['train_scores']]}"
            )
            print(
                f"Média T: {model['train_mean']:.5f} "
                f"+- {model['train_std']:.5f}"
            )

            print(
                f"Validação: F1 por fold: "
                f"{[f'{v:.5f}' for v in model['val_scores']]}"
            )
            print(
                f"Média V: {model['val_mean']:.5f} "
                f"+- {model['val_std']:.5f}"
            )

            print(
                f"Diferenças abs........: "
                f"{[f'{v:.5f}' for v in model['differences']]}"
            )
            print(
                f"Média das Diferenças..: {model['dif_mean']:.5f}\n"
                f"DPadrão das diferenças: {model['dif_std']:.5f}"
            )
        
    def choose_best_result(self):
        max_f1 = max(model['val_mean'] for model in self.models)
        
        candidates = [
            model 
            for model in self.models 
            if model['val_mean'] >= (max_f1 - 0.005)
        ]

        self.best_result = sorted(
            candidates,
            key=lambda x: (x['val_std'], x['dif_mean'], -x['val_mean'])
        )[0]

    def show_best_results(self):
        print(f"\n=============== Melhor modelo - {self.__class__.__name__.upper()} ===============")
        print("Melhor classificador Scikit")
        print(f"Parâmetros ..............: {self.clf.best_params_}")
        print(f"Média F1 de validação....: {self.clf.best_score_:.5f}")
        
        print("\nMelhor modelo")
        print(f"Parâmetros: {self.best_result['params']}")
        print(f"Validação média de F1........: {self.best_result['val_mean']:.5f}")
        print(f"Validação desvio padrão......: {self.best_result['val_std']:.5f}")
        print(f"Média das difs |treino-valid|: {self.best_result['dif_mean']:.5f}")