from Models.Model import Model
import xgboost as xgb

# GB Tree
class R_1(Model):
    """Model used for submission"""
    def __init__(self):
        super().__init__()
        self.model = self._init_model()

    def _init_model(self):
        """
        Model "finetuned" for hyperparameters
        """
        return xgb.XGBClassifier(
            n_estimators     = 1600,
            max_depth        = 16,
            learning_rate    = 0.10,
            verbosity        = 0,
            objective        = "mulit:softmax",
            booster          = 'gbtree',
            gamma            = 1,
            subsample        = 0.8,
            colsample_bytree = 0.8
        )
