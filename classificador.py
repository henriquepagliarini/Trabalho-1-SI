import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class Classificador:
    def __init__(self, dataset, model, parameters, n_folds):
        self.dataset = dataset
        self.model = model
        self.parameters = parameters
        self.n_folds = n_folds

        self.results = {}
        self.best_result = None
        self.best_estimator = None
        
    def train(self):
        self.clf = GridSearchCV(
            self.model,
            self.parameters,
            cv=self.n_folds,
            scoring='f1_macro',
            return_train_score=True
        )

        self.clf.fit(self.dataset.Xm_train, self.dataset.Ym_train)

        self.results = self.clf.cv_results_
    
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
        print(f"\n=============== Classificador {self.__class__.__name__} ===============")
        for i, model in enumerate(self.models):

            print(f"Modelo {i + 1}")
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
                f"DPadrão das diferenças: {model['dif_std']:.5f}\n"
            )
        
    def choose_best_result(self):
        self.best_estimator = self.clf.best_estimator_
        print(f"\n=============== Classificador {self.__class__.__name__} ===============")
        print("\n* Melhor classificador Scikit *")
        print(f"Parâmetros ..............: {self.clf.best_params_}")
        print(f"Média F1 de validação....: {self.clf.best_score_:.5f}")

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
        
        print("\n* Melhor modelo *")
        print(f"Parâmetros: {self.best_result['params']}")
        print(f"Validação média de F1........: {self.best_result['val_mean']:.5f}")
        print(f"Validação desvio padrão......: {self.best_result['val_std']:.5f}")
        print(f"Média das difs |treino-valid|: {self.best_result['dif_mean']:.5f}")
        
    def test_model(self):
        print(f"\n=============== Classificador {self.__class__.__name__} ===============")
        y_pred_train = self.best_estimator.predict(self.dataset.Xm_train)
        acc_train = accuracy_score(self.dataset.Ym_train, y_pred_train) * 100
        print(f"\nAcuracia com dados de treino: {acc_train:.2f}%")

        y_pred_test = self.best_estimator.predict(self.dataset.Xm_test)
        acc_test = accuracy_score(self.dataset.Ym_test, y_pred_test) * 100
        print(f"Acuracia com dados de teste: {acc_test:.2f}%")