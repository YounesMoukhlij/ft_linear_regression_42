from typing import List, Tuple
from colorama import init, Fore, Style
init(autoreset=True)


def main():
    km = []
    price = []
    try:
        km, price = parse_file(km, price)
        print("km ->", km)
        print("price ->", price)
        Slope, intercept = ft_linear_regression(km, price)
        test_data(Slope, intercept, km)
        estimateFunction(Slope, intercept)
    except FileNotFoundError as e:
        print(e)
    except BufferError as e:
        print(e)
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)


def estimateFunction(Slope, intercept):
    entred_mileage = float(input("\nEnter The Mileage of The car : "))
    print(f"{Fore.CYAN}Estimated Price for {Fore.YELLOW} Miles {entred_mileage} {Fore.RED} → {Fore.GREEN}{Slope * entred_mileage + intercept:.2f}{Style.RESET_ALL}")

def parse_file(km: List[float], price: List[float]) -> Tuple[List[float], List[float]]:
    with open("../data.csv", "r", encoding="utf-8") as file:
        for i in file:
            if not i:
                continue
            i = i.strip()
            if i.replace(",","",1).strip().isdigit():
                parts = i.split(",")
                if len(parts) < 2:
                    raise IndexError("list index out of range")
                km.append(float(parts[0].strip()))
                price.append(float(parts[1].strip()))
    return km, price


def ft_linear_regression(km: List[float], price: List[float]) -> Tuple[int, int ,int]:
    data_len = len(km)
    sum_x = 0
    sum_y = 0
    for i,j in zip(km, price):
        sum_x += i
        sum_y += j
    x_mean = sum_x / data_len
    y_mean = sum_y / data_len

    print(f"{Fore.CYAN}→  Data length : {data_len} values by items{Style.RESET_ALL}")
    print("\t\t    -- ")
    print(f"{Fore.YELLOW}→  Means are {Fore.BLUE}km{Style.RESET_ALL} : {Fore.GREEN}{x_mean:.2f}{Style.RESET_ALL}  && price : {Fore.MAGENTA}{y_mean:.2f}{Style.RESET_ALL}")
    print("\t\t    -- ")

    numerator = sum((i - x_mean) * (j - y_mean) for i,j in zip(km, price))
    denumerator = sum((k - x_mean)**2 for k in km)
    if denumerator == 0:
        raise ValueError("Could not Divide by zero")
    Slope = numerator / denumerator
    intercept = y_mean - Slope * x_mean
    print(f"{Style.BRIGHT}{Fore.MAGENTA}→  Slope:{Style.RESET_ALL} {Fore.GREEN}{Slope:.4f}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.MAGENTA}→  Intercept:{Style.RESET_ALL} {Fore.YELLOW}{intercept:.4f}{Style.RESET_ALL}")

    return Slope, intercept


def test_data(Slope: float, intercept : float, km :List[float]):
    estimatePrice = [Slope * x + intercept for x in km]
    print("Estimated values for the 5th giving in the  dataset : ")
    for x, y_hat in zip(km[:5], estimatePrice[:5]):
        print(f"→ km={x} {Fore.GREEN} < ~~~~~~ > {Fore.RESET} \
              estimate ={y_hat:.2f}")


if __name__ == "__main__":
    main()
