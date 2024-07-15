# Create a simulation of a bank account.

# The account should have a balance, a name and an account number.
# The account should have a method to withdraw money.
# The account should have a method to deposit money.
# The account should have a method to print the current balance.
# You need to design the class and implement the methods. Write tests for the methods for all possible scenarios.

class bank:
    def __init__(self, balance, name, account_number):
        self.bal = balance
        self.n = name
        self.an = account_number

    def withdraw(self, ammount):
        self.bal =- ammount

    def deposit(self, ammount):
        self.bal =+ ammount
        

    def display(self):
        return (self.bal)

# while True:
    
#     li = input("Hi there, Welcome to MIT Bank. Choose one of the following options. \n1. Login 2. Sign Up")
#     if li == 1:
#         if input("What is your name?") == user.n:
#             tt = input("Feel free to proceed with anny of the followinng:\n1. Withdraw\n2. Deposit\n3. Display Current Balance")
#             if tt == 1:
#                 user.withdraw(input("How much would you like to withdraw?"))
#             elif tt == 2:
#                 user.deposit(input("How much would you like to deposit?"))
#             elif tt == 3:
#                 print(user.display())
#             else:
#                 print("Invalid Input")
    


#     elif li == 2:
#         n = input("Welcome to MIT Bank. Please provide a name. ")
#         user = bank(0, n, 0)

#     else:
#         print("Invalid Input")

