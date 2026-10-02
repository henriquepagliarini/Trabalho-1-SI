from dataset import DataSet
from cart import Cart
from rn_mlp import Rn

def main():
    print("Lendo dados...")
    dataset = DataSet()
    dataset.load_data()
    dataset.split_data()

    print("Treinando CART...")
    cart = Cart(dataset, 5)
    cart.train()
    cart.process_results()
    
    print("Treinando RN MLP...")
    rn = Rn(dataset, 5)
    rn.train()
    rn.process_results()
    
    cart.show_results()
    rn.show_results()
    
    cart.choose_best_result()
    rn.choose_best_result()

    cart.test_model()
    rn.test_model()
    
if __name__ == '__main__':
    main()