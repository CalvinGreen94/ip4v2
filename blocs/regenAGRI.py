# regenAgriBlockchain.py

import uuid
import json
from datetime import datetime
import hashlib
import hmac
import os
import time
# -------------------------------
# Blockchain Core Structures
# -------------------------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# simple HMAC signer for demo (in prod use proper PKI)
BRIDGE_SECRET = os.environ.get("REGEN_BRIDGE_SECRET", "regen-demo-secret").encode()

def sign_message(msg: str) -> str:
    return hmac.new(BRIDGE_SECRET, msg.encode(), hashlib.sha256).hexdigest()

def verify_signature(msg: str, signature: str, secret: bytes = None) -> bool:
    sec = secret or BRIDGE_SECRET
    expected = hmac.new(sec, msg.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
class Block:
    def __init__(self, index, previous_hash, data):
        self.index = index
        self.timestamp = datetime.utcnow().isoformat()
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.generate_hash()

    def generate_hash(self):
        # deterministic hash from fields
        body = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True, separators=(",", ":"))
        return sha256_hex(body)

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis = Block(0, "0", {"message": "Genesis Block - RegenAgriBlockchain"})
        self.chain.append(genesis)







    def add_block(self, data):
        prev_block = self.chain[-1]
        new_block = Block(len(self.chain), prev_block.hash, data)
        self.chain.append(new_block)
        
        return new_block




    def export_chain(self):
        return [block.__dict__ for block in self.chain], 

# -------------------------------
# Regenerative Agriculture Context
# -------------------------------
class Farmer:
    def __init__(self, name, practice):
        self.name = name
        self.practice = practice  # e.g. cover cropping, rotational grazing
        self.inventory = {}

    def harvest(self, crop, qty):
        self.inventory[crop] = self.inventory.get(crop, 0) + qty
        return {"farmer": self.name, "crop": crop, "qty": qty, "practice": self.practice}

class Cooperative:
    def __init__(self, name):
        self.name = name

    def aggregate(self, deliveries):
        total = {}
        for d in deliveries:
            total[d["crop"]] = total.get(d["crop"], 0) + d["qty"]
        return {"cooperative": self.name, "aggregated": total}

class Buyer:
    def __init__(self, name):
        self.name = name
        self.stock = {}

    def purchase(self, crop, qty):
        self.stock[crop] = self.stock.get(crop, 0) + qty
        return {"buyer": self.name, "crop": crop, "qty": qty}

# -------------------------------
# Kanban / Tanpin-Kanri Simulation
# -------------------------------
class KanbanSystem:
    def __init__(self, threshold):
        self.threshold = threshold
        self.signals = []

    def check_reorder(self, buyer, crop):
        stock = buyer.stock.get(crop, 0)
        if stock < self.threshold:
            signal = {"kanban": f"Reorder triggered for {crop}", "time": datetime.utcnow().isoformat()}
            self.signals.append(signal)
            return signal
        return None

# -------------------------------
# Main Simulation
# -------------------------------
def simulate_regen_agri():
    blockchain = Blockchain()
    kanban = KanbanSystem(threshold=50)

    # Farmers practicing regenerative agriculture
    f1 = Farmer("Calvin", "cover cropping")
    f2 = Farmer("Grace", "rotational grazing")

    # Cooperative
    coop = Cooperative("Regen Farmers Co-op")

    # Buyer (organic food company)
    buyer = Buyer("G&G Foods Inc.")

    # Step 1: Farmers harvest
    harvests = [f1.harvest("wheat", 70), f2.harvest("corn", 60)]
    for h in harvests:
        blockchain.add_block({"event": "harvest", "details": h})

    # Step 2: Cooperative aggregates
    agg = coop.aggregate(harvests)
    blockchain.add_block({"event": "aggregation", "details": agg})

    # Step 3: Buyer purchases
    purchases = [buyer.purchase("wheat", 40), buyer.purchase("corn", 30)]
    for p in purchases:
        blockchain.add_block({"event": "purchase", "details": p})

    # Step 4: Kanban check
    for crop in buyer.stock:
        signal = kanban.check_reorder(buyer, crop)
        if signal:
            blockchain.add_block({"event": "kanban_signal", "details": signal})

    # Export chain
    return blockchain.export_chain()

# -------------------------------
# Run Simulation
# -------------------------------
if __name__ == "__main__":
    while True:
        time.sleep(3)
        chain_data = simulate_regen_agri()
        print(json.dumps(chain_data, indent=2))
