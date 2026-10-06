from dataset import DataSet
from cart import Cart
from rn import Rn
import matplotlib.pyplot as plt

def main():
    dataset = DataSet('./datasets/vict/10000v/data.csv')
    dataset.load_data()

    print("Treinando CART...")
    cart = Cart(dataset, 5)
    cart.train()
    cart.process_results()
    cart.show_results()
    cart.show_best_model()
    print("=================================================")
    
    print("\nTreinando RN...")
    rn = Rn(dataset, 5)
    rn.train()
    rn.process_results()
    rn.show_results()
    rn.show_best_model()
    
    cart.retrain()
    rn.retrain()
    
    print("\nSalvando CART...")
    cart.save_model()
    print("\nSalvando RN...")
    rn.save_model()
    
    test_dataset = DataSet('./datasets/vict/1300v/data.csv')
    test_dataset.load_data()
    
    cart.test(test_dataset)
    rn.test(test_dataset)
    plt.show()
    
    print("testando joblib")
    
    #Testando joblibs
    #------- REMOVER DEPOIS -------
    cart.test_joblib('melhor_cart.joblib', test_dataset)
    rn.test_joblib('melhor_rn.joblib', test_dataset)

if __name__ == '__main__':
    main()