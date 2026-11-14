import auth


def test_demo_credentials_and_signed_token():
    assert auth.authenticate(auth.USERNAME, auth.PASSWORD)
    assert not auth.authenticate(auth.USERNAME, "definitely-not-the-password")

    token = auth.create_access_token(auth.USERNAME)
    assert auth.verify_access_token(token) == auth.USERNAME
    assert auth.verify_access_token(token + "tampered") is None
