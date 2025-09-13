#pqc/crypto_utils.py
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
import base58  # pip install base58

def generate_pqc_keys():
    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key
    return verify_key.encode(encoder=HexEncoder), signing_key.encode(encoder=HexEncoder)

def sign_message(private_key_bytes, message_bytes):
    # private_key_bytes must be bytes, raw private key
    signing_key = SigningKey(private_key_bytes)  # no encoder here
    signed = signing_key.sign(message_bytes)
    return signed.signature.hex()


def verify_signature(public_key_base58: str, message_bytes: bytes, signature_hex: str) -> bool:
    verify_key_bytes = base58.b58decode(public_key_base58)
    verify_key = VerifyKey(verify_key_bytes)  # no encoder param, raw bytes
    try:
        verify_key.verify(message_bytes, bytes.fromhex(signature_hex))
        return True
    except Exception:
        return False