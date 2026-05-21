    
    # Train Model

from sklearn.linear_model import LogisticRegression
x= [[1]]
y= [1]
model = LogisticRegression()

model.fit(x, y)
    
    # Save Model

import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)          # saves as object