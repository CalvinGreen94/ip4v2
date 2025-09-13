# test_signing.py
from pqc.crypto_utils import generate_pqc_keys, sign_message, verify_signature

def test_fake_signed_message(ip_address):
    public_key, private_key = generate_pqc_keys()
    message = f"ALERT: Port scan detected on 192.168.1.157"

    signature = sign_message(private_key, message)
    verified = verify_signature(public_key, message, signature)

    return {
        "message": message,
        "signature": signature,
        "verified": verified
    }
