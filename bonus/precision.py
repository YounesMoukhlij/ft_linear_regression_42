import gradientDescent
import math

def main():
    km = []
    price = []

    # 1. Load the data
    # We reuse the parsing function from your existing file
    length = gradientDescent.parseFunction(km, price)

    if length == 0:
        print("Error: No data found.")
        return

    # 2. Get the trained model parameters
    # We retrain here to get the exact theta values your algorithm produces
    print("Training model to calculate precision...")
    theta0, theta1 = gradientDescent.train(km, price, length)
    print("\nCalculating statistics...")

    # 3. Calculate R-squared (Coefficient of Determination)
    # R^2 represents the proportion of the variance for the dependent variable (Price)
    # that's explained by the independent variable (Km).

    mean_price = sum(price) / length

    # Total Sum of Squares (Variance of the actual data)
    ss_tot = sum((y - mean_price) ** 2 for y in price)

    # Residual Sum of Squares (Variance of the prediction error)
    ss_res = sum((y - (theta0 + theta1 * x)) ** 2 for x, y in zip(km, price))

    r_squared = 1 - (ss_res / ss_tot)

    # 4. Calculate other metrics
    mse = ss_res / length
    rmse = math.sqrt(mse)

    # 5. Display results
    print("------------------------------------------------")
    print(f"Theta0 (Intercept)      : {theta0:.4f}")
    print(f"Theta1 (Slope)          : {theta1:.4f}")
    print("------------------------------------------------")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Sq Error(RMSE): {rmse:.2f}")
    print(f"R-squared Score         : {r_squared:.4f}")
    print("------------------------------------------------")

    if r_squared >= 0.9:
        print("Result: Excellent precision.")
    elif r_squared >= 0.7:
        print("Result: Good precision.")
    else:
        print("Result: Low precision. Consider adjusting learning rate or epochs.")

if __name__ == "__main__":
    main()
