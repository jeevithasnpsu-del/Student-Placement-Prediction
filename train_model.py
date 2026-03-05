import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

print("Starting model training...")

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully")

data['ExtracurricularActivities'] = data['ExtracurricularActivities'].map({'Yes':1,'No':0})
data['PlacementTraining'] = data['PlacementTraining'].map({'Yes':1,'No':0})
data['PlacementStatus'] = data['PlacementStatus'].map({'Placed':1,'NotPlaced':0})

X = data[['CGPA','Internships','Projects','Workshops/Certifications',
          'AptitudeTestScore','SoftSkillsRating',
          'ExtracurricularActivities','PlacementTraining',
          'SSC_Marks','HSC_Marks']]

y = data['PlacementStatus']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

print("Model trained successfully")

pickle.dump(model, open("model.pkl","wb"))

print("Model saved successfully")