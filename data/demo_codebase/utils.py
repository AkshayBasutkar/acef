"""
Demo utility functions.

Provides common helper functions used across the demo application,
including password hashing and input validation.
"""

import hashlib


def hash_password(pw: str) -> str:
    """
    Hash a plaintext password using MD5.

    Note: MD5 is used here for demo simplicity only.
    A real application should use bcrypt, scrypt, or argon2.

    Parameters
    ----------
    pw : str
        The plaintext password to hash.

    Returns
    -------
    str
        The hexadecimal MD5 digest of the password.
    """
    return hashlib.md5(pw.encode()).hexdigest()


def validate_username(username: str) -> bool:
    """
    Validate that a username meets basic requirements.

    Rules:
      - Must be between 3 and 20 characters
      - Must contain only alphanumeric characters and underscores

    Parameters
    ----------
    username : str
        The username to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if len(username) < 3 or len(username) > 20:
        return False
    return username.replace("_", "").isalnum()
