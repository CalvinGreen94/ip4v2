from pqc.crypto_utils import generate_pqc_keys, sign_message, verify_signature
from ml.ids_model import IntrusionDetector
from utils.data_loader import load_network_data
from scanner.nmap_scanner import run_nmap_scan
from openai import OpenAI
import flask 
import json
from web3 import Web3
import asyncio
import httpx
import csv
import os 
import time
from flask import Flask, session, abort, request, jsonify, render_template, redirect, url_for, flash, redirect, Response
import os
# import wikipedia
import datetime
import hashlib
from urllib.parse import urlparse
from flask_cors import CORS
import requests
from flask_bootstrap import Bootstrap
import openai
from flask_sslify import SSLify
from sklearn.metrics import mean_squared_error
from ta.momentum import RSIIndicator
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from uuid import *
import joblib
import threading
import subprocess
import openai
from uniswap import Uniswap
import qrcode
import datetime as dt
import logging
import json 

class IPqSECnet:

    def __init__(self):
        self.chain = []
        self.transactions = []
        self.create_block(proof=1, previous_hash='0000',
                            # pred_liq="0",
                            # pred_price="0", 
                            # pred_rsi="0",
                            # pred_price_change="0",
                            # pair="0",
                            # liq="0",
                            ) #,price="0",style="0",department="0" ,shipping_month="0"
        self.nodes = set()

    def create_block(self, proof, previous_hash,): #,price,style,department,shipping_month
  
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash': previous_hash,

                 }
        self.transactions = []
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while check_proof is False:
            hash_operation = hashlib.sha256(
                str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(
                str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
        return True

    # sender,receiver,amount # sender = sender receiver = receiver amount = amount
    def add_transaction(self,sender,receiver,amount):
        previous_block = blockchain.get_previous_block()
        previous_proof = previous_block['proof']
        proof = blockchain.proof_of_work(previous_proof)
        previous_hash = blockchain.hash(previous_block)
        self.transactions.append({
            'sender': sender,
            'receiver':receiver,
            'amount':amount,
        })
        previous_block = self.get_previous_block()
        return previous_block['index'] + 1

    def add_node(self, address):
        # address = 'http:127.0.0.1:8677/'
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
        # node = parsed_url.
# Give the Chain a Reason to exist

    def replace_chain(self):
        network = self.nodes
        longest_chain = None
        max_length = len(self.chain)
        for node in network:
            response = requests.get(f'http://{node}/get_chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.is_chain_valid(chain):
                    max_length = length
                    longest_chain = chain
        if longest_chain:
            self.chain = longest_chain
            return True
        return False



app = Flask(__name__)
sslify = SSLify(app)
bootstrap = Bootstrap(app)
blockchain = IPqSECnet()

node_address = str(uuid4()).replace('-', '') #New
root_node = 'e36f0158f0aed45b3bc755dc52ed4560d' #New
print("🔐 Generating post-quantum key pair...")
root_node, private_key = generate_pqc_keys()
print(f'public key: \n {root_node}, \n private key: \n {private_key}')
node_encrypted = hashlib.sha256(
                str(root_node).encode()).hexdigest()

print(node_encrypted)


def save_anamoly_data_to_json(data, filename='anomaly.json'):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)



def read_data_from_json(file_name):
    try:
        with open(file_name, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_updated_data_to_json(data, file_name="updated_data.json"):
    existing_data = read_data_from_json(file_name)
    existing_data.append(data)
    with open(file_name, "w") as file:
        json.dump(existing_data, file, indent=4)

def save_block_data_to_json(data, file_name="block_data.json"):
    existing_data = read_data_from_json(file_name)
    existing_data.append(data)
    with open(file_name, "w") as file:
        json.dump(existing_data, file, indent=4)
def extract_features_from_scan(scan_results):
    features_list = []
    for entry in scan_results:
        # If warning present, create a zero-filled feature vector
        if 'warning' in entry:
            features_list.append({
                'tcp_open': 0,
                'tcp_closed': 0,
                'udp_open': 0,
                'udp_closed': 0,
                'ssh_present': 0,
                'http_present': 0
            })
            continue

        tcp_ports = entry.get('tcp', {})
        udp_ports = entry.get('udp', {})

        features = {
            'tcp_open': sum(1 for p in tcp_ports if tcp_ports[p]['state'] == 'open'),
            'tcp_closed': sum(1 for p in tcp_ports if tcp_ports[p]['state'] == 'closed'),
            'udp_open': sum(1 for p in udp_ports if udp_ports[p]['state'] == 'open'),
            'udp_closed': sum(1 for p in udp_ports if udp_ports[p]['state'] == 'closed'),
            'ssh_present': int(any(tcp_ports[p]['name'] == 'ssh' and tcp_ports[p]['state'] == 'open' for p in tcp_ports)),
            'http_present': int(any(tcp_ports[p]['name'] == 'http' and tcp_ports[p]['state'] == 'open' for p in tcp_ports))
        }
        features_list.append(features)
    return pd.DataFrame(features_list)




# from utils.fake_data_generator import generate_fake_intrusion_data
# from utils.fake_nmap_results import generate_fake_nmap_results
# from test_signing import test_fake_signed_message
import json


@app.route('/hash/<hash_value>')
def display_hash(hash_value):
    # Assuming you have a function to retrieve hash details
    data = blockchain.chain
    hash_details = next((item for item in data if item.get('previous_hash') == hash_value), None)

    if hash_details:
        return render_template('hash_details.html', hash=hash_details)

    # If hash is not found
    return render_template('hash_details.html', error="Hash information not found.")

@app.route('/is_valid', methods = ['GET'])
def is_valid():
    is_valid = blockchain.is_chain_valid(blockchain.chain)
    message = {} 
    data = {}
    if is_valid:
        # response = {'message': 'All good. The Blockchain is valid.'}
        message = 'All good,Blockchain Is Valid' 
        # data['status'] = 200 
        # data['data'] = message
    else:
        response = {'message': 'Houston, we have a problemo. The Blockchain is not valid.'}
        message['message'] = 'Houston, we have a problemo. The Blockchain is not valid' 
        data['status'] = 200 
        data['data'] = message   
    return jsonify(data)

@app.route('/add_transaction', methods = ['POST'])
def add_transaction():
    message = {} 
    data = {}
    json = request.get_json()
    transactions_keys= ['sender','receiver','amount']
    if not all (key in json for key in transactions_keys):
        message['message'] = 'HOME ELMENTS OF THE TRASACTION ARE MISSING' 
        data['status'] =  400
        data['data'] = message   
        return jsonify(data) #'HOME ELMENTS OF THE TRASACTION ARE MISSING' 
    index = blockchain.add_transaction(json['sender'],json['receiver'],json['amount']) 
    response = {'message': f'This Transaction IS NOW ON BLOCK {index}'}
    message['message'] = 'This Transaction IS NOW ON BLOCK {}'.format(index)
    data['status'] = 201 
    data['data'] = message   
    return jsonify(response),201

### Decentralizing the Network 

###Connecting Nodes 
@app.route('/connect_node',methods=["POST"]) 
def connect_node():
    received_json = request.get_json() 
    nodes = received_json.get('nodes')
    print(f'NODES: {nodes}')
    if nodes is None:
        message = ' No Node Found'
        return render_template('connected.html',nodes=nodes,message = message)
    for node in nodes:
        blockchain.add_node(node)
        message = 'All the nodes are now connected. The Blockchain now contains the following nodes:'
        total_nodes= list(blockchain.nodes)

    # data['status'] = 201 
    # data['data'] = message   
    return render_template('index.html',nodes=nodes,connected = message,total_nodes=total_nodes)


### Connect longest chain if necessary
@app.route('/replace_chain', methods = ['GET'])
def replace_chain():
    is_chain_replaced = blockchain.replace_chain()
    message = {} 
    data = {}
    if is_chain_replaced:
        response = {'message': 'NODES HAD DIFFERENT CHAINS , REPLACED BY LONGEST CHAIN',
        'new_chain': blockchain.chain }
        message['message'] = 'NODES HAD DIFFERENT CHAINS , REPLACED BY LONGEST CHAIN {}'.format(blockchain.chain)
        data['status'] = 200 
        data['data'] = message
    else:
        response = {'message': 'NODE IS CONNECT TO LARGEST CHAIN',
        'actual_chain':blockchain.chain}
        message['message'] = 'NODE IS CONNECT TO LARGEST CHAIN {}'.format(blockchain.chain)
        data['status'] = 200 
        data['data'] = message   
    return jsonify(data)







usr_agent = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}


from datetime import datetime

def log_event(private_key, event_data, log_path="logs.json"):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_data
    }
    log_json = json.dumps(log_entry, sort_keys=True)
    signature = sign_message(private_key, log_json)
    signed_log = {
        "log": log_entry,
        "signature": signature
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(signed_log) + "\n")


def verify_logs(public_key, log_path="logs.json"):
    from pqc.crypto_utils import verify_signature

    with open(log_path, "r") as f:
        for line in f:
            signed_log = json.loads(line)
            log_data = json.dumps(signed_log["log"], sort_keys=True)
            signature = signed_log["signature"]

            if verify_signature(public_key, log_data, signature):
                print("[✅ Valid] ", signed_log["log"])
            else:
                print("[❌ Tampered] ", signed_log["log"])


def serialize_message(data):
    return json.dumps(data, sort_keys=True).encode('utf-8')

def main():
    while True:
        try:
            print("🔐 Generating post-quantum key pair...")
            public_key, private_key = generate_pqc_keys()
            print(f'public key: \n {public_key}, \n private key: \n {private_key}')

            print("📡 Running Nmap scan...")
            scan_results = run_nmap_scan()  # Replace with subnet or IP range
            print(f"Scan Results: {json.dumps(scan_results[:5], indent=2)}")

            # results = run_nmap_scan(num_hosts=3)
            # for r in results:
            #     print("\n🔎 Scan Summary for Host:", r["host"])
            #     for v in r["vulnerabilities"]:
            #         print(f" - Port {v['port']} ({v['protocol']}): {v['product']} {v['version']} - CVEs: {v['cve_output'][:60]}...")



            # print("📦 Loading synthetic intrusion detection dataset...")
            print("🔧 Extracting features for IDS training...")
            
            X = extract_features_from_scan(scan_results)
            print(X)


            print("🧠 Training IDS model...")
            ids = IntrusionDetector()
            # ids.train(X)

            X_synthetic = load_network_data()
            ids.train(X_synthetic)

            # Then fine-tune or re-train on real scan data
            X_real = extract_features_from_scan(scan_results)
            ids.train(X_real)

            



            print("🔎 threat prediction...")
            predictions = ids.predict(X_real)
            print(predictions)

            time.sleep(5)
            for idx, pred in enumerate(predictions):
                if pred == -1:
                    report["ids_alerts"].append({
                        "index": idx,
                        "status": "Anomalous",
                        "confidence": round(ids.decision_function([X.iloc[idx]])[0], 3)
                    })
                    print(pred)
                # log_event(private_key.decode())

                # save_anamoly_data_to_json(private_key.decode())
            

            print("✍️ Signing Nmap scan data...")
            message_bytes = serialize_message(scan_results[:5])
            signature = sign_message(private_key, message_bytes)
            is_valid = verify_signature(public_key, message_bytes, signature)
            print(signature)
            timestamp = datetime.utcnow().isoformat()
            report = {
                "timestamp": timestamp,
                # "target": target,
                "scan_results": scan_results,
                "ids_alerts": [],
                "pqc_signature": None,
            }
            print(report)
            signed = sign_message(json.dumps(report, indent=2), private_key)
            report["pqc_signature"] = signed
            print(report)


            # is_valid = verify_signature(public_key, serialized, signature)

            print("✅ Verifying scan result signature...")
            is_valid1 = verify_signature(public_key, message, signature)

            print(f"Signature Valid: {is_valid1}")
            print("📶 PQC-IDS system with Nmap integration is operational.")
            print(f'IDS QS: n {signature}')
            print(f'predictions: \n {predictions}')


            # logging.basicConfig(level=logging.INFO)
            # logger = logging.getLogger(__name__)
            # logger.info("Starting IDS system...")

            # intrusion_df = generate_fake_intrusion_data()
            # nmap_data = generate_fake_nmap_results()
            # signed_data = test_fake_signed_message()

            # print("\n=== Intrusion Data Sample ===")
            # print(intrusion_df.head())

            # print("\n=== Fake Nmap Results ===")
            # print(json.dumps(nmap_data, indent=2))

            # print("\n=== Signed Alert ===")
            # print(json.dumps(signed_data, indent=2))
            # time.sleep(5) 

            previous_block = blockchain.get_previous_block()
            previous_proof = previous_block['proof']
            proof = blockchain.proof_of_work(previous_proof)
            previous_hash = blockchain.hash(previous_block)

            trans = blockchain.add_transaction(sender=root_node, receiver=node_address, amount=1.15)

            # message_to_sign = serialize_message(nmap_data)
            # signature = sign_message(private_key, message_to_sign)
            signed_message_obj = {
                "message": json.loads(message_bytes),  # readable
                "signature": signature,
                "public_key": public_key
            }

            # block = blockchain.create_block(
            #     proof, previous_hash,
            #     intrusion_data=intrusion_df.to_dict(orient="records"),
            #     nmap_results=nmap_data,
            #     signed_message=signed_message_obj
            # )


            block = blockchain.create_block(proof, previous_hash, 
                                        intrusion_data=scan_results,
                                        nmap_results=X_real,
                                        signed_message=signed_message_obj
                                            ) #price=price,style=style,department=department,shipping_month=shipping_month, **file_paths
                            #price=price,style=style,department=department,shipping_month=shipping_month, **file_paths
            print(trans)
            print(block)
            save_block_data_to_json(block)
            message = 'Congratulations, you just mined Snope Block {} at {}! Proof of work {}, previous hash {}\n, block {}, transactions: {}'.format(
                                block['index'], block['timestamp'], block['proof'], block['previous_hash'], block, block['transactions'])

            is_chain_replaced = blockchain.replace_chain()

            if is_chain_replaced:
                chain_replaced = 'NODES HAD DIFFERENT CHAINS, REPLACED BY LONGEST CHAIN'
            else:
                chain_replaced = 'NODE IS CONNECTED TO LARGEST CHAIN'

            is_valid = blockchain.is_chain_valid(blockchain.chain)
            print(block)
            full_chain()

            if is_valid:
                valid = 'All good, Blockchain Is Valid'
            else:
                valid = 'Houston, we have a problemo. The Blockchain is not valid'
                    # return jsonify(balances_to_return)
        except Exception as e:
            message = f"Error occurred: {e}"

def full_chain():
    fullChain = 'full blockchain {}, {}'.format(len(blockchain.chain),blockchain.chain)
    print(fullChain)
    return fullChain

    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #         messages=[
    #             {"role": "system", "content": "You are a cybersecurity analyst assistant that explains output from an IDS and Nmap scanner in plain language."},
    #             {"role": "user", "content": f"The signature verification {signature} ,What does this mean?"}
    #         ],
    #     temperature=0.7,
    #     max_tokens=500,
    # )
    # return response.choices[0].message.content.strip()
if __name__ == "__main__":
    time.sleep(10)
    main()
