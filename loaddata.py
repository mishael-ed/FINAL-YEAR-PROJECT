import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_excel(r"C:\Users\misha\OneDrive\Desktop\Student Academic Predictor\student_data.xlsx", sheet_name = "Student Records")

#problem w the raw data is that the model only understands numbers
#therefore i will use numbers to encode the words like "good" "excellent"

behavioral_map = {
    "Poor": 1,
    "Average":2,
    "Good": 3,
    "Excellent":4
}

grade_map = {
    "F": 1,  
    "E": 2,
    "D": 3,
    "C": 4,
    "B": 5,
    "A": 6  
}

outcome_map = {
    "Fail": 0,
    "Pass": 1
}

df["Behavioral_Enc"] = df["Behavioral Rating"].map(behavioral_map)
df["Grade_Enc"]      = df["Grade"].map(grade_map)
df["Outcome_Enc"]    = df["Final Outcome"].map(outcome_map)





#feature engr
#here, every student has 18 rows in the table. I want every student to have one row each instead

all_students = []

for student_id, group in df.groupby("Student ID"):
    
    
    row = {}
    row["Student ID"]    = student_id
    row["avg_total"]     = group["Total Score"].mean()
    row["min_total"]     = group["Total Score"].min()
    row["avg_ca"]        = group["CA Score"].mean()
    row["avg_attendance"]= group["Attendance %"].mean()
    row["num_fails"]     = (group["Outcome_Enc"] == 0).sum()
    row["fail_rate"]     = row["num_fails"] / len(group)
    
    # TARGET: what are we trying to predict?
    # 1 = this student had at least one fail (at risk)
    # 0 = this student passed everything (safe)
    row["at_risk"] = 1 if row["num_fails"] > 0 else 0
    
    all_students.append(row)

#actual training 

features_df = pd.DataFrame(all_students)

X = features_df.drop(columns=["Student ID", "at_risk"])
y = features_df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing = 40 students
    stratify=y,         # preserve class balance
    random_state=42     # so you get same split every time
)

rf = RandomForestClassifier(
    n_estimators=200,        # build 200 trees
    class_weight="balanced", # because we have unequal at-risk vs safe
    random_state=42
)

rf.fit(X_train, y_train)   # learning

print("Training done!")

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, 
      target_names=["Safe", "At-Risk"]))


#LSTM
df = pd.read_excel(r"C:\Users\misha\OneDrive\Desktop\Student Academic Predictor\student_data.xlsx",
                   sheet_name="Student Records")

df["BE"] = df["Behavioral Rating"].map({"Poor":1,"Average":2,"Good":3,"Excellent":4})
df["GE"] = df["Grade"].map({"F":1,"E":2,"D":3,"C":4,"B":5,"A":6})
df["OE"] = df["Final Outcome"].map({"Fail":0,"Pass":1})
df["TN"] = df["Term"].map({"2022_T1":1,"2022_T2":2,"2022_T3":3})

#here the student data is grouped by the student id and the term number... the rows are a summary of a students data for that particular term
term_records = []

for (student_id, term_num), group in df.groupby(["Student ID", "TN"]):
    row = {}
    row["Student ID"]  = student_id
    row["Term_Num"]    = term_num
    row["avg_ca"]      = group["CA Score"].mean()
    row["avg_exam"]    = group["Exam Score"].mean()
    row["avg_total"]   = group["Total Score"].mean()
    row["min_total"]   = group["Total Score"].min()
    row["avg_att"]     = group["Attendance %"].mean()
    row["min_att"]     = group["Attendance %"].min()
    row["avg_beh"]     = group["BE"].mean()
    row["avg_grade"]   = group["GE"].mean()
    row["fail_count"]  = (group["OE"] == 0).sum()
    row["fail_rate"]   = row["fail_count"] / len(group)
    row["at_risk"]     = 1 if row["fail_count"] > 0 else 0
    term_records.append(row)



#sequences 

FEATURES = ["avg_ca","avg_exam","avg_total","min_total",
            "avg_att","min_att","avg_beh","avg_grade","fail_rate"] #here we define the input variables thta the model will use

SEQ_LEN = 2   # use 2 terms to predict the 3rd

X_sequences = [] #this will be the sequences for features
y_labels    = [] #this will hold the target labels

term_df = pd.DataFrame(term_records).sort_values(["Student ID", "Term_Num"])
for student_id, group in term_df.groupby("Student ID"):
    group = group.sort_values("Term_Num") #sorting the terms in the correct order so that the model doesnt learn wrong patterns
    
    feature_values = group[FEATURES].values   # shape: (3, 9)
    risk_labels    = group["at_risk"].values  # shape: (3,)
    
    # sliding window
    for i in range(len(feature_values) - SEQ_LEN):
        X_sequences.append(feature_values[i : i + SEQ_LEN])  # terms 1 & 2
        y_labels.append(risk_labels[i + SEQ_LEN])             # term 3 label

X = np.array(X_sequences)  # shape: (200, 2, 9)
y = np.array(y_labels)     # shape: (200,)

print("X shape:", X.shape)   # (200, 2, 9) — 200 samples, 2 timesteps, 9 features
print("y shape:", y.shape)   # (200,)
print("At-risk:", y.sum())



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# fit scaler only on training data
scaler = StandardScaler()

# have to reshape to 2D to fit, then reshape back to 3D
X_train_flat = X_train.reshape(-1, 9)  
X_test_flat  = X_test.reshape(-1, 9)    

scaler.fit(X_train_flat)   # learn mean and std from training data only

X_train = scaler.transform(X_train_flat).reshape(X_train.shape)
X_test  = scaler.transform(X_test_flat).reshape(X_test.shape)

model = Sequential([
    LSTM(64, input_shape=(2, 9), return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1,  activation="sigmoid")   # sigmoid = outputs 0 to 1 probability
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",   # standard loss for 0/1 classification
    metrics=["accuracy"]
)

model.summary()   # prints the architecture



history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,        # how many times to go through the data
    batch_size=16,    # how many samples per update
    verbose=1
)




y_prob = model.predict(X_test).flatten()  # probability scores
y_pred = (y_prob >= 0.5).astype(int)      # convert to 0 or 1

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Safe", "At-Risk"]))