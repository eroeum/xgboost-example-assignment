from Models.Model import Model
import xgboost as xgb

# Attempted Random Forest Model
class ModelRF(Model):
    def __init__(self):
        super().__init__()
        self.model = self._init_model()

    def _init_model(self):
        return xgb.XGBRFClassifier()
