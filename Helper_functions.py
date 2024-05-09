from sklearn.metrics import r2_score
from sklearn.model_selection import KFold



def KFold_function(model, n_splits: int):
  scores = []
  KF = KFold(n_splits=n_splits)
  for train_index, val_index in KFold.split(X_train):
      X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
      y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
      func_model = model.fit(X_train_fold, y_train_fold)
        
      preds = func_model.predict(X_val_fold)
        
      score = r2_score(y_val_fold, preds)
      scores.append(score)
