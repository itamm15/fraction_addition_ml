import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = {
    "fraction1_numerator": [1, 1, 2, 1, 1],
    "fraction1_denominator": [2, 4, 3, 2, 4],
    "fraction2_numerator": [1, 1, 1, 1, 1],
    "fraction2_denominator": [3, 4, 6, 3, 4],
    "user_answer_numerator": [2, 2, 5, 3, 2],
    "user_answer_denominator": [5, 8, 6, 6, 8],
    # Typy błędów: 0 = poprawne, 1 = dodanie liczników i mianowników oddzielnie, 2 = błąd wspólnego mianownika, 3 = błąd w licznikach
    "error_type": [1, 0, 0, 2, 3]
}

df = pd.DataFrame(data)
print(df)

# Dane wejściowe (ułamki i odpowiedzi użytkownika)
X = df[["fraction1_numerator", "fraction1_denominator", "fraction2_numerator", "fraction2_denominator", "user_answer_numerator", "user_answer_denominator"]]
# Etykiety, które zawierają typ błędu
y = df["error_type"]

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu klasyfikującego typ błędu
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Przewidywanie typów błędów dla zbioru testowego
predictions = model.predict(X_test)

# Wyświetlanie wyników
print("Przewidywane typy błędów:", predictions)
print("Rzeczywiste typy błędów:", y_test.values)

new_data = pd.DataFrame({
    "fraction1_numerator": [1, 2],
    "fraction1_denominator": [2, 3],
    "fraction2_numerator": [1, 2],
    "fraction2_denominator": [3, 3],
    "user_answer_numerator": [3, 6],
    "user_answer_denominator": [6, 6]
})

# Przewidywanie typu błędu dla nowego przykładu
predicted_error = model.predict(new_data)
print("Przewidywany typ błędu:", predicted_error[0])

# Opcjonalnie: mapowanie przewidywanego typu błędu na opis
error_types = {
    0: "Poprawna odpowiedź",
    1: "Dodano licznik i mianownik osobno",
    2: "Błąd we wspólnym mianowniku",
    3: "Błąd w przekształceniu liczników"
}

print("Opis błędu:", error_types[predicted_error[0]])
