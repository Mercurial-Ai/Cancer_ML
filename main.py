from src.data_pipeline import data_pipeline

class cancer_ml:
    def __init__(self, dataset, target, model="clinical_only"):
        self.dataset = dataset
        self.target = target
        self.model = model

    def collect_duke(self):
        self.data_pipe = data_pipeline("data\\Duke-Breast-Cancer-MRI", "", self.target)
        self.data_pipe.load_data()
    
    def collect_HN1(self):
        self.data_pipe = data_pipeline("data\\HNSCC-HN1", "")
        