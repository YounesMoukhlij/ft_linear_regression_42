import estimatePrice
import  gradientDescent
from colorama import init, Fore, Back, Style


if __name__ == "__main__":

    print(f"\t{Fore.GREEN}---> {Fore.RESET} {Fore.RED} First Program : Simple Linear Regression {Fore.RESET} {Fore.GREEN} <---{Fore.RESET}\n")
    estimatePrice.main()
    print(f"\n\t{Fore.GREEN}---> {Fore.RESET} {Fore.RED} Second Program : Gradient Descent {Fore.RESET} {Fore.GREEN} <---{Fore.RESET}\n")
    gradientDescent.main()
