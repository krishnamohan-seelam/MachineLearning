
import  xgboost as xgb 

class XgBoostWrapper(object):
    def __init__(self,seed,params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)
    
    def train(self,x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)
    
    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))