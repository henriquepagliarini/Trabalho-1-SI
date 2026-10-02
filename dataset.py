import pandas as pd

from sklearn.model_selection import train_test_split

class DataSet:
    def __init__(self):
        self.data_path = './datasets/vict/10000v/data.csv'

        self.df = None

        self.Xm = None
        self.Ym = None

        self.Xm_train = None
        self.Xm_test = None

        self.Ym_train = None
        self.Ym_test = None

        self.test_size = 0.30
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)

        self.Xm = self.df[['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']]
        self.Ym = self.df['tri']

    def split_data(self):
        self.Xm_train, self.Xm_test, self.Ym_train, self.Ym_test = train_test_split(
            self.Xm, 
            self.Ym, 
            test_size=self.test_size, 
            random_state=42, 
            shuffle=True, 
            stratify=self.Ym
            )