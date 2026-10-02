from classificador import Classificador
from sklearn.tree import DecisionTreeClassifier

class Cart(Classificador):
    def __init__(self, dataset, n_folds):
        parameters = [
            {
                # Subajustada (U)
                'criterion': ['entropy'],
                'max_depth': [2],
                'min_samples_leaf': [50]
            },
            {
                # Equilibrada (E)
                'criterion': ['entropy'],
                'max_depth': [8],
                'min_samples_leaf': [10],
            },
            {
                # Sobreajustada (O)
                'criterion': ['entropy'],
                'max_depth': [30],
                'min_samples_leaf': [1]
            }
        ]

        model = DecisionTreeClassifier(random_state=42)
        
        super().__init__(
            dataset,
            model,
            parameters,
            n_folds
        )