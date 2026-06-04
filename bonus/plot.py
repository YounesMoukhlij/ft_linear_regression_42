import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mandatory'))
import gradientDescent


def calculatePrice(km: list[float], slope: float, intercept: float) -> list[float]:
    """Calculate estimated prices using the gradient descent model."""
    y = []
    for x in km:
        y.append(slope * x + intercept)
    return y

def main() -> None:
    with open('../data.csv', 'r') as file:
        price = []
        km = []
        for line in file:
            try:
                tmpKm, tmpPrice = line.strip().split(',')
                km_val = float(tmpKm)
                price_val = float(tmpPrice)
                price.append(price_val)
                km.append(km_val)
            except (ValueError, IndexError):
                continue

        finalPrice = np.array(price, dtype='float64')
        finalKm = np.array(km, dtype='float64')
        print("Data loaded:")
        print(f"km: {finalKm[:5]}...")
        print(f"price: {finalPrice[:5]}...")

        theta0, theta1 = gradientDescent.train(finalKm.tolist(), finalPrice.tolist(), len(finalKm))

        estimatedPrice = calculatePrice(finalKm.tolist(), theta0, theta1)


        estimatedPrice_array = np.array(estimatedPrice, dtype='float64')
        sorted_indices = np.argsort(finalKm)
        sorted_km = finalKm[sorted_indices]
        sorted_actual_price = finalPrice[sorted_indices]
        sorted_estimated_price = estimatedPrice_array[sorted_indices]

        plt.figure(figsize=(30, 10))

        plt.subplot(1, 2, 1)
        plt.scatter(finalKm, finalPrice, alpha=0.5, label='Actual Data')
        plt.xlabel("Distance (km)")
        plt.ylabel("Price")
        plt.title("Data Distribution")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(sorted_km, sorted_actual_price, alpha=0.5, label='Actual Data', color='blue')
        plt.plot(sorted_km, sorted_estimated_price, 'r-', linewidth=2, label='Gradient Descent Fit')
        plt.xlabel("Distance (km)")
        plt.ylabel("Price")
        plt.title("Linear Regression: Gradient Descent Model")
        plt.legend()
        plt.grid(True)

        print(f"\nModel parameters:")
        print(f"Theta0 (intercept): {theta0:.4f}")
        print(f"Theta1 (slope): {theta1:.4f}")

        plt.show()

if __name__ == "__main__":
    main()
