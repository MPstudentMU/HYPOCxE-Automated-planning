"""Tests for engine/auth.py's shared-passphrase hashing/verification."""
from __future__ import annotations

import hashlib

from engine.auth import hash_passphrase, verify_passphrase


def test_hash_passphrase_matches_hand_computed_sha256():
    # Hand-computed, not captured from hash_passphrase's own output.
    assert hash_passphrase("hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_hash_passphrase_matches_stdlib_hashlib_directly():
    assert hash_passphrase("clinic2026") == hashlib.sha256(b"clinic2026").hexdigest()


def test_hash_passphrase_is_deterministic():
    assert hash_passphrase("same input") == hash_passphrase("same input")


def test_hash_passphrase_is_case_and_whitespace_sensitive():
    assert hash_passphrase("Secret") != hash_passphrase("secret")
    assert hash_passphrase("secret") != hash_passphrase("secret ")


def test_verify_passphrase_correct():
    expected = hash_passphrase("clinic2026")
    assert verify_passphrase("clinic2026", expected) is True


def test_verify_passphrase_incorrect():
    expected = hash_passphrase("clinic2026")
    assert verify_passphrase("wrong-guess", expected) is False


def test_verify_passphrase_empty_input_does_not_match_a_real_hash():
    expected = hash_passphrase("clinic2026")
    assert verify_passphrase("", expected) is False


def test_verify_passphrase_returns_bool_not_truthy_object():
    expected = hash_passphrase("clinic2026")
    assert verify_passphrase("clinic2026", expected) is True
    assert verify_passphrase("nope", expected) is False
