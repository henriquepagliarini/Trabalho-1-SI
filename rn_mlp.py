from classificador import Classificador
from sklearn.neural_network import MLPClassifier

class Rn(Classificador):
    def __init__(self, dataset, n_folds):
        parameters = [
            {
                # Subajustada (U)
                'hidden_layer_sizes': [(50, 25, 5)],
                'activation': ['relu'],
                'learning_rate_init': [0.010]
            },
            {
                # Equilibrada (E)
                'hidden_layer_sizes': [(4, 4, 8, 8, 8)],
                'activation': ['relu'],
                'learning_rate_init': [0.003]
            },
            {
                # Sobreajustada (O)
                'hidden_layer_sizes': [(100, 50)],
                'activation': ['relu'],
                'learning_rate_init': [0.01]
            }
        ]
        
        model = MLPClassifier(random_state=42, max_iter=250)

        super().__init__(
            dataset,
            model,
            parameters,
            n_folds
        )