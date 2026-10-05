import pandas as pd

class DataSet:
    def __init__(self, data_path):
        self.data_path = data_path

        self.df = None

        self.Xm = None
        self.Ym = None
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)

        self.Xm = self.df[['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']]
        self.Ym = self.df['tri']