import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ufc_gold_dataset_final.csv")

print("Dimenzije dataseta:", df.shape)
print("\nPrvih 5 redova:")
print(df.head())
print("\nKolone:")
print(df.columns.tolist())
print("Distribucija načina pobjede:")
print(df["Method"].value_counts())
print("\nProcentualno:")
print(df["Method"].value_counts(normalize=True) * 100)
print("Nedostajuće vrijednosti po koloni:")
print(df.isnull().sum())# Grupisanje Method u 3 glavne klase
def group_method(method):
    if method in ["Decision - Unanimous", "Decision - Split", "Decision - Majority"]:
        return "Decision"
    elif method in ["KO/TKO", "TKO - Doctor's Stoppage"]:
        return "KO/TKO"
    elif method == "Submission":
        return "Submission"
    else:
        return None  # izbaci rijetke klase

df["Method_Grouped"] = df["Method"].apply(group_method)

# Izbaci redove sa None
df = df.dropna(subset=["Method_Grouped"])

print("Nakon čišćenja:")
print(df["Method_Grouped"].value_counts())
print(f"\nUkupno borbi: {len(df)}")

features = [
    "F1_Sig_Landed",   # broj značajnih udaraca borca 1
    "F2_Sig_Landed",   # broj značajnih udaraca borca 2
    "F1_TD_Landed",    # broj uspješnih rušenja borca 1
    "F2_TD_Landed",    # broj uspješnih rušenja borca 2
    "F1_Ctrl_Sec",     # kontrola na podu borca 1 (sekunde)
    "F2_Ctrl_Sec",     # kontrola na podu borca 2 (sekunde)
    "F1_Sub_Att",      # pokušaji gušenja borca 1
    "F2_Sub_Att",      # pokušaji gušenja borca 2
]

X = df[features]
y = df["Method_Grouped"]

print("Dimenzije X:", X.shape)
print("Dimenzije y:", y.shape)
print("\nPrvih 5 redova X:")
print(X.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Podjela 80/20, stratifikovano
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train skup: {X_train.shape[0]} borbi")
print(f"Test skup: {X_test.shape[0]} borbi")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nRezultati Random Forest:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Skaliranje (za Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Rezultati Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

importances = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("Važnost varijabli:")
print(importances)

# Grafikon
plt.figure(figsize=(10, 6))
plt.barh(importances["Feature"], importances["Importance"], color="steelblue")
plt.xlabel("Važnost")
plt.title("Važnost varijabli za predikciju načina pobjede")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("slike/feature_importance.png")
plt.show()

# Confusion matrix za Random Forest
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred, labels=["KO/TKO", "Submission", "Decision"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["KO/TKO", "Submission", "Decision"],
            yticklabels=["KO/TKO", "Submission", "Decision"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.tight_layout()
plt.savefig("slike/confusion_matrix.png")
plt.show()

# Distribucija načina pobjede
plt.figure(figsize=(8, 5))
df["Method_Grouped"].value_counts().plot(kind="bar", color=["steelblue", "coral", "seagreen"])
plt.title("Distribucija načina pobjede")
plt.xlabel("Način pobjede")
plt.ylabel("Broj borbi")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("slike/method_distribution.png")
plt.show()

# Boxplot udaraca
plt.figure(figsize=(10, 6))
sns.boxplot(x="Method_Grouped", y="F1_Sig_Landed", data=df, palette="Set2")
plt.title("Broj značajnih udaraca po načinu pobjede")
plt.xlabel("Način pobjede")
plt.ylabel("Značajni udarci (borac 1)")
plt.tight_layout()
plt.savefig("slike/sig_landed_boxplot.png")
plt.show()