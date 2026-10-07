"""
Demo authentication module.

Provides a login function that validates user credentials
by checking the password hash against stored hashes.
"""

from utils import hash_password


# Simulated user database
_USER_DB = {
    "admin": "5f4dcc3b5aa765d61d8327deb882cf99",
    "alice": "d8578edf8458ce06fbc5bb76a58c5ca4",
}


def login(username: str, password: str) -> bool:
    """
    Authenticate a user by username and password.

    Hashes the provided password and compares it against the stored
    hash in the user database.

    Parameters
    ----------
    username : str
        The username to authenticate.
    password : str
        The plaintext password to verify.

    Returns
    -------
    bool
        True if authentication succeeds, False otherwise.
    """
    if username not in _USER_DB:
        return False
    expected_hash = _USER_DB[username]
    actual_hash = hash_password(password)
    return actual_hash == expected_hash


def create_user(username: str, password: str) -> bool:
    """
    Create a new user account.

    Parameters
    ----------
    username : str
        Desired username (must not already exist).
    password : str
        Plaintext password to store (will be hashed).

    Returns
    -------
    bool
        True if user was created, False if username already exists.
    """
    if username in _USER_DB:
        return False
    _USER_DB[username] = hash_password(password)
    return True
