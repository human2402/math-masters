import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# --- Настройка ---
np.random.seed(42)  # для воспроизводимости

dim = 3  # размерность (x1, x2, x3)
bounds = np.array([[0.0, 1.0]] * dim)  # область D: xi ∈ [0,1]

def g(x):
    x1, x2, x3 = x
    val = 0.8 * (x1 - 0.8)**4 + 0.4 * (x2 - 0.4)**2 + 0.4 * (x3 - 0.1)**2 + 5.0
    return math.log(val)

# целевая функция для максимизации 
def target(x):
    return -g(x)

# --- Параметры ---
S = 30        # число разведчиков
N = 6         # лучшие точки
M = 6         # перспективные точки
L = 10        # пчёлы для уточнения вокруг лучших
P = 4         # пчёлы для уточнения вокруг перспективных
R0 = 0.25     # начальный радиус окрестности
R_min = 0.001  # минимальный радиус
alpha_R = 0.98 # уменьшение радиуса на каждой итерации
max_iters = 200

# генерация случайных точек в окрестности
def sample_neighborhood(center, R, n_samples):
    dim = len(center)
    samples = center + (np.random.rand(n_samples, dim) * 2 - 1) * R
    for i in range(dim):
        samples[:, i] = np.clip(samples[:, i], bounds[i,0], bounds[i,1])
    return samples

# объединение пересекающихся окрестностей
def merge_neighborhoods(centers, values, R):
    centers = centers.copy()
    values = values.copy()
    S = len(centers)
    merged = []
    taken = np.zeros(S, dtype=bool)
    for i in range(S):
        if taken[i]:
            continue
        group = [i]
        for j in range(i+1, S):
            if np.linalg.norm(centers[i]-centers[j]) < R:
                group.append(j)
        best_idx = max(group, key=lambda k: values[k])
        merged.append((centers[best_idx], values[best_idx]))
        for k in group:
            taken[k] = True
    new_centers = np.array([m[0] for m in merged])
    new_vals = np.array([m[1] for m in merged])
    return new_centers, new_vals

# --- Инициализация ---
X = np.random.rand(S, dim) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
vals = np.array([target(x) for x in X])

best_history = []
best_point = X[np.argmax(vals)]
best_val = np.max(vals)
R = R0

history_rows = []

# --- Основной цикл ---
for it in range(max_iters):
    order = np.argsort(-vals)
    best_indices = order[:N]
    promising_indices = order[N:N+M]
    new_centers = []
    new_vals = []
    
    # уточнение вокруг лучших
    for idx in best_indices:
        center = X[idx]
        samples = sample_neighborhood(center, R, L)
        sample_vals = np.array([target(s) for s in samples])
        max_idx = np.argmax(np.concatenate(([vals[idx]], sample_vals)))
        if max_idx == 0:
            new_centers.append(center)
            new_vals.append(vals[idx])
        else:
            new_centers.append(samples[max_idx-1])
            new_vals.append(sample_vals[max_idx-1])
    
    # уточнение вокруг перспективных
    for idx in promising_indices:
        center = X[idx]
        samples = sample_neighborhood(center, R, P)
        sample_vals = np.array([target(s) for s in samples])
        max_idx = np.argmax(np.concatenate(([vals[idx]], sample_vals)))
        if max_idx == 0:
            new_centers.append(center)
            new_vals.append(vals[idx])
        else:
            new_centers.append(samples[max_idx-1])
            new_vals.append(sample_vals[max_idx-1])
    
    # оставшиеся центры без изменений
    remaining = [i for i in range(S) if i not in np.concatenate((best_indices, promising_indices))]
    for idx in remaining:
        new_centers.append(X[idx])
        new_vals.append(vals[idx])
    
    new_centers = np.array(new_centers)
    new_vals = np.array(new_vals)
    
    # объединяем пересекающиеся окрестности
    new_centers, new_vals = merge_neighborhoods(new_centers, new_vals, R)
    
    X = new_centers
    vals = new_vals
    S = len(X)
    
    # обновляем глобальный максимум
    it_best_idx = np.argmax(vals)
    if vals[it_best_idx] > best_val:
        best_val = vals[it_best_idx]
        best_point = X[it_best_idx].copy()
    best_history.append(best_val)
    
    # сохраняем историю
    history_rows.append({
        "iter": it,
        "S": S,
        "R": R,
        "best_val": best_val,
        "best_x1": best_point[0],
        "best_x2": best_point[1],
        "best_x3": best_point[2],
    })
    
    # уменьшаем радиус
    R = max(R * alpha_R, R_min)
    
    if S <= 1 or R <= R_min:
        break

# --- Вывод ---
history_df = pd.DataFrame(history_rows)
print(history_df.head(10))

print("\nНайденная точка (x1,x2,x3):", best_point)
print("Значение целевой функции (максимум):", best_val)
print("Число итераций:", len(best_history))

# график сходимости
plt.figure(figsize=(8,4))
plt.plot(best_history)
plt.title("Сходимость: максимум target по итерациям")
plt.xlabel("итерация")
plt.ylabel("best target value")
plt.grid(True)
plt.show()
