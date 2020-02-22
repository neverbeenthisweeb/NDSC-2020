# Lecturer:
# Zaky Syihan
# (https://www.linkedin.com/in/ahmad-zaky-syihan-53014b174/)

# 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# accuracy_score -->
# confusion_matrix --> Melihat kecenderungan model ML untuk prediksi ke arah mana TP TN FP FN
# classification_report --> (precision, recall, f1-score)


# 2
data = pd.read_csv("data/syn.csv")
data.head()


# 3
X = data[["amount"]]
y = data["isFraud"]


# 4
lr = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
lr.fit(X_train, y_train)
lr.predict(X_test)


# 5
# Parameter tuning --> (hyperparameter, learnable parameter)


# 6
# Ensemble method:
#   Voting --> (hard voting, soft voting)


# 7
# Bonus tips!
#   Do a lot of experiment
#   Copy (and learn...) people's works
#   Master DS workflow
#   Learn EDA
#   Learn maths


# 8
# To win this competition
#   Understand what makes data dirty
#   Learn how to handle dirty data
#   Explore state-of-the-art models
#   Try to be scientific  in every step
#   RULES OF THUMBS!
