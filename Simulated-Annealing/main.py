'''
    МЕТОД ИМИТАЦИИ ОТЖИГА. ЛАБОРАТОРНАЯ РАБОТА
    Вариант 10
    Выпролнил Малежик Д. ИВТм-25

    Реализуйте метод имитации отжига и найдите точку минимума указанной функции. Точность вычислений задавайте сами.
'''

import numpy as np
import pandas as pd

def f(x):
    x1, x2, x3 = x
    return np.log(0.8*(x1 - 0.8)**4 + 0.4*(x2 - 0.4)**2 + 0.4*(x3 - 0.1)**2 + 5)


def simulated_annealing_with_table(
        f, x0,
        T_start=1.0,
        T_min=0.00001,
        alpha=0.99,
        step_size=0.1,
        print_every=100
    ):
    x = np.array(x0)
    T = T_start
    f_val = f(x)

    iteration = 0
    history = []   # данные для таблицы

    while T > T_min:
        iteration += 1
        x_new = x + step_size * np.random.uniform(-1, 1, size=len(x))
        f_new = f(x_new)
        delta = f_new - f_val

        accepted = False
        step_type = ""

        if delta < 0:
            x, f_val = x_new, f_new
            accepted = True
            step_type = "лучше"
        else:
            p = np.exp(-delta / T)
            if np.random.rand() < p:
                x, f_val = x_new, f_new
                accepted = True
                step_type = "хуже, принято"
            else:
                step_type = "отброшено"

        # Сохраняем в таблицу
        history.append({
            "iter": iteration,
            "температура": T,
            "x1": x[0],
            "x2": x[1],
            "x3": x[2],
            "f(x)": f_val,
            "принято": accepted,
            "вид_шага": step_type
        })

        # Вывод какждый N-ый шаг
        if iteration % print_every == 0:
            print(f"{iteration:5d} | Delta={delta:5f} | T={T:.5f} | f={f_val:.6f} | x={x} | {step_type}")

        T *= alpha

    # Convert to table
    df = pd.DataFrame(history)
    return x, f_val, df


# RUN
x0 = [0, 0, 0]
xmin, fmin, table = simulated_annealing_with_table(f, x0)

print("\nИтоговый резултат:")
print("Наименьшая точка:", xmin)
print("Значение функции:", fmin)

print("\nТаблица:")
print(table.head(20))
print('...')
print(table.tail(20))
