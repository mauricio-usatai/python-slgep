import numpy as np

from slgep import C_SLGEP


model = C_SLGEP(2)
model.generate_random_population(50)

X = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

model.set_dataset(X, y)

chromosome, fitness = model.population[0]
prediction = model.predict(np.array([1]), chromosome)
print(prediction)

history = model.fit(10)
print(history)
