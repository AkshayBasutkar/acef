"""
Demo application entry point.

A simple CLI that demonstrates the login flow using the auth module.
"""

from auth import login


def main():
    """
    Run the demo application.

    Prompts the user for credentials and attempts to log them in.
    """
    print("=== Demo Login System ===")
    username = input("Username: ")
    password = input("Password: ")

    if login(username, password):
        print(f"Welcome, {username}!")
    else:
        print("Login failed. Invalid username or password.")


if __name__ == "__main__":
    main()
