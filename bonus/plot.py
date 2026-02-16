import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../mandatory'))
import estimatePrice as eP


def calculatePrice(price: list[float],slope: float, intercept: float) -> list[float]:
    y = []
    for x in price:
        y.append(slope * x + intercept)
    return y

def main() -> None:
    with open('../data.csv', 'r') as file:
        price = []
        km = []
        for line in file:
            tmpKm, tmpPrice = line.strip().split(',')
            if tmpPrice.isdigit() or tmpKm.isdigit():
                price.append(tmpPrice)
                km.append(tmpKm)
        finalPrice = np.array(price, dtype='int64')
        finalKm = np.array(km, dtype='int64')
        print(finalPrice)
        print(finalKm)
        sortedKm = np.sort(finalKm)

        # plot with the calculated slope and intercept from estimatePrice
        tmpSlope, tmpIntercept = eP.ft_linear_regression([float(x) for x in km ], [float(y) for y in price])

        estimatedPrice = calculatePrice([float(x) for x in km], tmpSlope, tmpIntercept)


        tmp = [int(x) for x in estimatedPrice]
        newPrice = np.array(tmp)


        plt.figure(figsize=(30, 10))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
        plt.plot(finalPrice, finalKm)
        plt.xlabel("Price")
        plt.ylabel("Distance (km)")
        plt.title("Not Sorted")

        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
        plt.plot(finalPrice, sortedKm, 'bo', label='Actual Data')
        plt.plot(newPrice, sortedKm, 'r-', label='Prediction')
        plt.xlabel("Price")
        plt.ylabel("Distance (km)")
        plt.title("Price vs Distance")
        plt.legend()
        plt.show()
    return None

if __name__ == "__main__":
    main()
