import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

with open("data.pickle", "rb") as f:
    data_dict = pickle.load(f)

data = []
labels = []

for x, y in zip(data_dict["data"], data_dict["labels"]):
    if len(x) == 42:
        data.append(x)
        labels.append(y)

data = np.asarray(data)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
model.fit(x_train, y_train)

print("Doğruluk:", model.score(x_test, y_test))
from sklearn.utils.multiclass import unique_labels

labels_used = unique_labels(y_test, model.predict(x_test))
print(classification_report(y_test, model.predict(x_test), labels=labels_used, target_names=le.inverse_transform(labels_used)))


with open("model.p", "wb") as f:
    pickle.dump({"model": model, "label_encoder": le}, f)
