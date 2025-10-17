
from typing import List, Tuple
import math

def parseFunction(km : List[float], price: List[float]) :
    try:
        with open("../data.csv", "r", encoding="utf-8") as f:

            line = f.readlines()

            for i in line:
                if not i:
                    continue
                data = i.split(",")

                try :
                    km.append(float(data[0].strip()))
                    price.append(float(data[1].strip()))
                except ValueError:
                    continue

    except ValueError as e:
        print(e)
    return len(km)

def ft_linear_regression(km: List[float], price: List[float], length: int) -> Tuple[float, float, float, float]:

    x_mean = sum(km) / length
    y_mean = sum(price) / length

    denominator =  sum((x - x_mean) ** 2 for x in km)


    if denominator == 0:
        raise ValueError("Denominator is zero: all km values are identical")


    Slope = sum((x - x_mean) * (y - y_mean) for x,y in zip(km, price)) / denominator
    Intercept = y_mean - Slope * x_mean


    return Slope, Intercept, x_mean, y_mean


def train(km: List[float], price: List[float], length: int) -> Tuple[float, float]:
    epochs = 2000
    lr = 1e-9

    theta0 = 0.0
    theta1 = 0.0

    for epoch in range(1, epochs + 1):
        # predictions
        yp = [theta0 + theta1 * xi for xi in km]

        # errors: prediction - actual
        error = [ypi - yi for ypi, yi in zip(yp, price)]

        # mse for monitoring and NaN detection
        mse = sum(e * e for e in error) / length
        # if math.isnan(mse) or math.isinf(mse):
        #     print(f"Stopped at epoch {epoch}: mse became {mse}; theta0={theta0}, theta1={theta1}")
        #     break

        # gradient steps (subject formulas)
        tmp0 = lr * (sum(error) / length)
        tmp1 = lr * (sum(er * x for er, x in zip(error, km)) / length)

        # # optional: clip updates to avoid runaway
        # tmp0 = max(min(tmp0, 1e6), -1e6)
        # tmp1 = max(min(tmp1, 1e6), -1e6)

        theta0 -= tmp0
        theta1 -= tmp1

        # periodic logging
        if epoch == 1 or epoch == epochs:
            print(f"epoch {epoch:6d}  mse={mse:.6f}  theta0={theta0:.6f}  theta1={theta1:.6f}")

    return theta0, theta1








def main():
    print("Gradient Descent")
    km = []
    price = []

    try:

        length = parseFunction(km, price)
        if length == 0:
            raise ValueError ("No training data provided")
        Slope, Intercept, x_mean, y_mean = ft_linear_regression(km, price, length)
        theta0, theta1 = train(km, price, length)
    except ValueError as e:
        print(e)





