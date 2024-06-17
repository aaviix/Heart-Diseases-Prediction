from PyQt5.QtWidgets import QApplication, QTabWidget, QLineEdit, QComboBox
from PyQt5.QtWidgets import QLabel, QPushButton, QSlider
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont, QIntValidator, QDoubleValidator
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('heart_disease_data_final.csv')
print(f'data shape: {data.shape}')

# Dividing data into X and Y
X = data.drop(columns='target', axis=1)
Y = data['target']

# Splitting the Data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2, stratify=Y)

# Model Training
model = LogisticRegression()
model.fit(x_train.values, y_train.values)

# accuracy
x_test_predict = model.predict(x_test)
x_test_accu = accuracy_score(x_test_predict, y_test)
print("Accuracy on test data: ", x_test_accu)


class Window(FigureCanvas):
    def __init__(self, parent):
        fig, ax = plt.subplots(2, figsize=(8.9, 10))
        super().__init__(fig)
        self.setParent(parent)

        sns.heatmap(data.corr(), annot=True, cmap='YlGnBu', ax=ax[0])
        pd.crosstab(data["age"], data["target"]).plot.bar(figsize=(5, 5), xlabel="Age", ylabel="Disease", ax=ax[1])


class AppMain(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(900, 900)
        Window(self)


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Disease Prediction")
        self.setFixedSize(QSize(1500, 1000))
        self.font = QFont("Arial", 10)

        self.head = QLabel("Please Enter Patient's details.", self)
        self.head.setFont(QFont("Arial", 12, QFont.Bold))
        self.head.move(900, 100)

        # all labels and inputs
        self.label_age = QLabel("Age:", self)
        self.label_age.move(900, 220)
        self.label_age.setFont(self.font)
        self.value_age = QSlider(Qt.Horizontal, self)
        self.value_age.setMinimum(20)
        self.value_age.setMaximum(100)
        self.value_age.setTickInterval(1)
        self.value_age.setSingleStep(1)
        self.value_age.move(1100, 215)
        self.value_age.resize(150, 30)

        self.value_age_label = QLabel("20", self)
        self.value_age_label.move(1300, 220)
        self.value_age_label.setFont(self.font)
        self.value_age.valueChanged.connect(self.update_value_age_label)

        self.label_sex = QLabel("Sex:", self)
        self.label_sex.move(900, 280)
        self.label_sex.setFont(self.font)
        self.value_sex = QComboBox(self)
        self.value_sex.addItem("Male")
        self.value_sex.addItem("Female")
        self.value_sex.move(1100, 275)
        self.value_sex.resize(150, 30)

        self.label_cp = QLabel("Chest Pain Type (0-4):", self)
        self.label_cp.move(900, 340)
        self.label_cp.setFont(self.font)
        self.value_cp = QLineEdit(self)
        self.value_cp.setValidator(QIntValidator(1, 1000, self))
        self.value_cp.move(1100, 340)
        self.value_cp.resize(150, 30)

        self.label_thalach = QLabel("Max Heart Rate (60-300):", self)
        self.label_thalach.move(900, 400)
        self.label_thalach.setFont(self.font)
        self.value_thalach = QLineEdit(self)
        self.value_thalach.setValidator(QIntValidator(1, 100, self))
        self.value_thalach.move(1100, 395)
        self.value_thalach.resize(150, 30)

        self.label_exang = QLabel("Exercise-induced Angina:", self)
        self.label_exang.move(900, 460)
        self.label_exang.setFont(self.font)
        self.value_exang = QComboBox(self)
        self.value_exang.addItem("Yes")
        self.value_exang.addItem("No")
        self.value_exang.move(1100, 460)
        self.value_exang.resize(150, 30)

        self.label_oldpeak = QLabel("ST Depression (0.0-7.0):", self)
        self.label_oldpeak.move(900, 520)
        self.label_oldpeak.setFont(self.font)
        self.value_oldpeak = QLineEdit(self)
        self.value_oldpeak.setValidator(QDoubleValidator(0.0, 10.0, 1, self))
        self.value_oldpeak.move(1100, 520)
        self.value_oldpeak.resize(150, 30)

        self.label_slope = QLabel("ST segment slope (0-2):", self)
        self.label_slope.move(900, 580)
        self.label_slope.setFont(self.font)
        self.value_slope = QLineEdit(self)
        self.value_slope.setValidator(QIntValidator(1, 10, self))
        self.value_slope.move(1100, 580)
        self.value_slope.resize(150, 30)

        self.label_ca = QLabel("Blood vessels blockage (0-4):", self)
        self.label_ca.move(900, 640)
        self.label_ca.setFont(self.font)
        self.value_ca = QLineEdit(self)
        self.value_ca.setValidator(QIntValidator(1, 10, self))
        self.value_ca.move(1100, 640)
        self.value_ca.resize(150, 30)

        self.label_thal = QLabel("Thallium stress test (0-3):", self)
        self.label_thal.move(900, 700)
        self.label_thal.setFont(self.font)
        self.value_thal = QLineEdit(self)
        self.value_thal.setValidator(QIntValidator(1, 10, self))
        self.value_thal.move(1100, 700)
        self.value_thal.resize(150, 30)

        # submit button
        self.button = QPushButton("Submit", self)
        self.button.setCheckable(True)
        self.button.move(900, 760)
        self.button.setCheckable(True)
        self.button.clicked.connect(self.predict_disease)

        # output label
        self.prediction = QLabel("", self)
        self.prediction.setGeometry(900, 820, 300, 50)
        self.prediction.setFont(QFont("Arial", 10, QFont.Bold))

        self.tab = QWidget(self)
        self.addTab(main, "Prediction Model")

    def update_value_age_label(self, value):
        self.value_age_label.setText(str(value))

    def predict_disease(self):
        age = self.value_age.value()
        sex = self.value_sex.currentText()
        cp = self.value_cp.text()
        thalach = self.value_thalach.text()
        exang = self.value_exang.currentText()
        oldpeak = self.value_oldpeak.text()
        slope = self.value_slope.text()
        ca = self.value_ca.text()
        thal = self.value_thal.text()
        print(f'raw user input: {(age, sex, cp, thalach, exang, oldpeak, slope, ca, thal)}')

        if age and sex and cp and thalach and exang and oldpeak and slope and ca and thal:
            sex = (1 if sex == "Male" else 0)
            exang = (1 if exang == "Yes" else 0)
            user_input = [age, int(sex), int(cp), int(thalach), exang, float(oldpeak), int(slope), int(ca), int(thal)]
            print(f'user input: {user_input}')

            user_input = np.asarray(user_input)
            user_input = user_input.reshape(-1, 9)

            # Perform prediction
            predict = model.predict(user_input)
            print(f"Prediction: {predict}")

            if predict[0] == 0:
                self.prediction.setText("Patient doesn't have a heart disease :)")
            else:
                self.prediction.setText("Patient have a heart disease!!")
        else:
            self.prediction.setText("Please fill all the details!")


app = QApplication(sys.argv)
main = AppMain()
final = MainWindow()
main.show()
final.show()
sys.exit(app.exec_())
