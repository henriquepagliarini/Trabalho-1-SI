from classificador import Classificador
from sklearn.neural_network import MLPClassifier

class Rn(Classificador):
    def __init__(self, dataset, n_folds):
        parameters = [
            {
                # Subajustada (U)
                'hidden_layer_sizes': [(6, 2)],
                'activation': ['relu'],
                'learning_rate_init': [0.05],
                'solver': ['adam']
            },
            {
                # Equilibrada (E)
                'hidden_layer_sizes': [(16, 8)],
                'activation': ['relu'],
                'learning_rate_init': [0.01],
                'solver': ['adam']
            },
            {
                # Sobreajustada (O)
                'hidden_layer_sizes': [(100, 50, 25)],
                'activation': ['tanh'],
                'learning_rate_init': [0.001],
                'solver': ['adam']
            }
        ]
        
        model = MLPClassifier(random_state=42, max_iter=500)

        super().__init__(
            dataset,
            model,
            parameters,
            n_folds
        )