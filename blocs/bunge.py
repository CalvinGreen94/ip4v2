#!/usr/bin/env python3
"""
Bunge – Digital Commodity Settlement Network (Simulation)
With optional Post-Quantum signatures (liboqs) and an on-chain shipment lifecycle:

- PQ signature support: tries to use liboqs-python (oqs). If not available,
  falls back to the HMAC demo signer.
- Shipment lifecycle: dispatch -> in_transit updates -> customs_clearance -> receipt_confirmation
- Settlement (payment) occurs only after receipt_confirmation; reconciliation separately recorded.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib, hmac, json, uuid, os, sys, traceback

# ---------------------------------------------------------------------
# Utilities: hashing
# ---------------------------------------------------------------------
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ---------------------------------------------------------------------
# Signature abstraction: try PQ (liboqs) then fallback to HMAC
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Signature abstraction: try PQ (pqcrypto) then fallback to HMAC
# ---------------------------------------------------------------------
class BaseSigner:
    def __init__(self, name: str):
        self.name = name

    def sign(self, message: str) -> str:
        raise NotImplementedError

    def verify(self, message: str, signature: str) -> bool:
        raise NotImplementedError

# Try to import pqcrypto for PQ signatures
PQ_AVAILABLE = False
try:
    from pqcrypto.sign import dilithium2
    PQ_AVAILABLE = True
except ImportError:
    PQ_AVAILABLE = False

if PQ_AVAILABLE:
    class PQSigner(BaseSigner):
        """
        pqcrypto wrapper using Dilithium2 signatures.
        Keys are generated once for demo purposes; in production you'd persist keys.
        """
        def __init__(self, name: str, seed: Optional[bytes] = None):
            super().__init__(name)
            self.public_key, self.secret_key = dilithium2.generate_keypair()

        def sign(self, message: str) -> str:
            sig = dilithium2.sign(message.encode(), self.secret_key)
            return sig.hex()

        def verify(self, message: str, signature: str) -> bool:
            try:
                sig_bytes = bytes.fromhex(signature)
                # Extract message back from signature to verify
                msg_out = dilithium2.open(sig_bytes, self.public_key)
                return msg_out.decode() == message
            except Exception:
                return False
else:
    class HMACSigner(BaseSigner):
        """
        Demo fallback signer: HMAC-SHA256 (shared secret). Keeps same simple API.
        """
        def __init__(self, name: str, secret: Optional[str] = None):
            super().__init__(name)
            self.secret = (secret or sha256(uuid.uuid4().bytes)[:32]).encode()

        def sign(self, message: str) -> str:
            return hmac.new(self.secret, message.encode(), hashlib.sha256).hexdigest()

        def verify(self, message: str, signature: str) -> bool:
            expected = self.sign(message)
            return hmac.compare_digest(expected, signature)

# Helper factory
def create_signer(name: str, secret: Optional[str] = None):
    if PQ_AVAILABLE:
        return PQSigner(name)
    else:
        return HMACSigner(name, secret)


# ---------------------------------------------------------------------
# Permissioning & Supply Chain Compliance
# ---------------------------------------------------------------------
@dataclass
class Entity:
    id: str
    name: str
    role: str                # "validator" | "bunge_hq" | "plant" | "port" | "farmer" | "logistics" | "auditor" | "customs"
    signer: BaseSigner
    kyc_verified: bool = True
    aml_risk_score: int = 1                # For payments: 1 (low) .. 5 (high)
    supply_cert_verified: bool = True     # Supplier certificate (e.g., contract + traceability)
    quality_score: int = 90               # 0-100 quality measure for commodity batches
    phytosanitary_cert: bool = True       # Plant health / export clearance
    sustainability_score: int = 80        # ESG score (0-100)

class CertificateAuthority:
    def __init__(self):
        self.registry: Dict[str, Entity] = {}

    def enroll(self,
               name: str,
               role: str,
               secret: Optional[str]=None,
               kyc=True,
               aml_score=1,
               supply_cert=True,
               quality_score=90,
               phyto=True,
               sustainability_score=80) -> Entity:
        ent_id = str(uuid.uuid4())
        signer = create_signer(name, secret)
        entity = Entity(
            id=ent_id,
            name=name,
            role=role,
            signer=signer,
            kyc_verified=kyc,
            aml_risk_score=aml_score,
            supply_cert_verified=supply_cert,
            quality_score=quality_score,
            phytosanitary_cert=phyto,
            sustainability_score=sustainability_score
        )
        self.registry[ent_id] = entity
        return entity

    def get(self, ent_id: str) -> Entity:
        return self.registry[ent_id]

# ---------------------------------------------------------------------
# Transactions, Blocks, Policy
# ---------------------------------------------------------------------
@dataclass
class ShipmentEvent:
    event_id: str
    actor_id: str
    event_type: str  # dispatch | in_transit | customs_cleared | receipt_confirmed | reconciliation
    details: Dict
    timestamp: str
    signature: str

@dataclass
class SettlementTx:
    tx_id: str
    sender_facility_id: str
    receiver_facility_id: str
    amount_usd: int
    commodity_type: str
    batch_id: str
    weight_kg: float
    created_at: str
    metadata: Dict = field(default_factory=dict)
    approvals: Dict[str, str] = field(default_factory=dict)
    # lifecycle tracking
    lifecycle_status: str = "created"  # created, dispatched, in_transit, customs_cleared, receipt_confirmed, settled
    history: List[ShipmentEvent] = field(default_factory=list)

    def to_json(self) -> str:
        # canonical JSON for signing & hashing (exclude approvals & history for core intent)
        payload = {
            "tx_id": self.tx_id,
            "sender_facility_id": self.sender_facility_id,
            "receiver_facility_id": self.receiver_facility_id,
            "amount_usd": self.amount_usd,
            "commodity_type": self.commodity_type,
            "batch_id": self.batch_id,
            "weight_kg": self.weight_kg,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "lifecycle_status": self.lifecycle_status
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

@dataclass
class Block:
    index: int
    prev_hash: str
    timestamp: str
    txs: List[SettlementTx]
    proposer_id: str
    votes: Dict[str, str] = field(default_factory=dict)
    block_hash: str = ""

    def compute_hash(self) -> str:
        data = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "tx_ids": [t.tx_id for t in self.txs],
            "proposer_id": self.proposer_id,
            "votes": self.votes
        }
        return sha256(json.dumps(data, sort_keys=True).encode())

class Policy:
    def __init__(self,
                 high_value_threshold: int = 500_000,
                 max_aml_score: int = 3,
                 approvers_required: int = 2,
                 min_quality_score: int = 70,
                 min_sustainability_score: int = 60):
        self.high_value_threshold = high_value_threshold
        self.max_aml_score = max_aml_score
        self.approvers_required = approvers_required
        self.min_quality_score = min_quality_score
        self.min_sustainability_score = min_sustainability_score

    def requires_multisig(self, amount: int) -> bool:
        return amount >= self.high_value_threshold

# ---------------------------------------------------------------------
# Ledger, consensus, audit log, lifecycle management
# ---------------------------------------------------------------------
class AuditLog:
    def __init__(self, path="bunge_audit.log"):
        self.path = path
        self._prev_hash = "0"*64
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                lines = f.read().splitlines()
                if lines:
                    try:
                        last = json.loads(lines[-1].decode())
                        self._prev_hash = last.get("entry_hash", self._prev_hash)
                    except Exception:
                        pass

    def append(self, entry: Dict, signer: BaseSigner):
        entry_body = {
            "timestamp": datetime.utcnow().isoformat(),
            "prev_entry_hash": self._prev_hash,
            "entry": entry
        }
        entry_json = json.dumps(entry_body, sort_keys=True, separators=(",", ":"))
        entry_hash = sha256(entry_json.encode())
        signature = signer.sign(entry_json)
        record = {
            "entry": entry_body,
            "entry_hash": entry_hash,
            "signature": signature,
            "signed_by": signer.name
        }
        with open(self.path, "ab") as f:
            f.write((json.dumps(record) + "\n").encode())
        self._prev_hash = entry_hash

class Ledger:
    def __init__(self, validators: List[Entity], auditor: Entity, policy: Policy):
        self.validators = validators
        self.auditor = auditor
        self.policy = policy
        self.blocks: List[Block] = []
        genesis = Block(index=1, prev_hash="0"*64, timestamp=datetime.utcnow().isoformat(), txs=[], proposer_id="genesis")
        genesis.block_hash = genesis.compute_hash()
        self.blocks.append(genesis)
        self.audit = AuditLog()
        # local mempool for pending shipments (not yet on-chain settled)
        self.pending_shipments: Dict[str, SettlementTx] = {}

    @property
    def head(self) -> Block:
        return self.blocks[-1]

    def _bft_quorum(self) -> int:
        n = len(self.validators)
        return (2 * n) // 3 + 1

    def propose_block(self, proposer: Entity, txs: List[SettlementTx]) -> Block:
        block = Block(
            index=len(self.blocks)+1,
            prev_hash=self.head.block_hash,
            timestamp=datetime.utcnow().isoformat(),
            txs=txs,
            proposer_id=proposer.id
        )
        return block

    def vote_block(self, block: Block) -> Tuple[bool, Block]:
        header_str = json.dumps({
            "index": block.index,
            "prev_hash": block.prev_hash,
            "timestamp": block.timestamp,
            "tx_ids": [t.tx_id for t in block.txs],
            "proposer_id": block.proposer_id,
        }, sort_keys=True, separators=(",", ":"))

        votes = {}
        for v in self.validators:
            if self._validate_txs(block.txs):
                votes[v.id] = v.signer.sign(header_str)
        block.votes = votes
        if len(votes) >= self._bft_quorum():
            block.block_hash = block.compute_hash()
            self.blocks.append(block)
            self.audit.append({
                "event": "BLOCK_COMMIT",
                "block_index": block.index,
                "tx_ids": [t.tx_id for t in block.txs],
                "votes": list(votes.keys())
            }, signer=self.auditor.signer)
            # mark shipments as settled
            for t in block.txs:
                t.lifecycle_status = "settled"
                if t.tx_id in self.pending_shipments:
                    del self.pending_shipments[t.tx_id]
            return True, block
        else:
            return False, block

    # --------------------------
    # Validation: supply chain compliance, quality, multi-sig approvals
    # --------------------------
    def _validate_txs(self, txs: List[SettlementTx]) -> bool:
        for tx in txs:
            sender = self._find_entity(tx.sender_facility_id)
            receiver = self._find_entity(tx.receiver_facility_id)
            if sender is None or receiver is None:
                return False

            if not (sender.kyc_verified and receiver.kyc_verified):
                return False
            if sender.aml_risk_score > self.policy.max_aml_score:
                return False

            if not sender.supply_cert_verified:
                return False
            if not sender.phytosanitary_cert:
                return False
            if sender.quality_score < self.policy.min_quality_score:
                return False
            if sender.sustainability_score < self.policy.min_sustainability_score:
                return False

            submit_sig = tx.metadata.get("submitter_sig")
            submitter_id = tx.metadata.get("submitter_id")
            if not submit_sig or not submitter_id:
                return False
            submitter = self._find_entity(submitter_id)
            if submitter is None:
                return False
            if not submitter.signer.verify(tx.to_json(), submit_sig):
                return False

            if self.policy.requires_multisig(tx.amount_usd):
                if len(tx.approvals) < self.policy.approvers_required:
                    return False
                approved = 0
                for approver_id, sig in tx.approvals.items():
                    approver = self._find_entity(approver_id)
                    if approver and approver.signer.verify(tx.to_json(), sig):
                        approved += 1
                if approved < self.policy.approvers_required:
                    return False
        return True

    def _find_entity(self, ent_id: str) -> Optional[Entity]:
        for v in self.validators:
            if v.id == ent_id:
                return v
        if self.auditor.id == ent_id:
            return self.auditor
        return ENTITY_LOOKUP.get(ent_id)

    # --------------------------
    # Shipment lifecycle helpers
    # --------------------------
    def register_shipment(self, tx: SettlementTx):
        """Add a shipment to pending mempool and audit the registration."""
        self.pending_shipments[tx.tx_id] = tx
        self.audit.append({
            "event": "SHIPMENT_REGISTERED",
            "tx_id": tx.tx_id,
            "batch_id": tx.batch_id,
            "sender": tx.sender_facility_id,
            "receiver": tx.receiver_facility_id,
            "amount_usd": tx.amount_usd
        }, signer=self.auditor.signer)

    def _append_event(self, tx: SettlementTx, actor: Entity, event_type: str, details: Dict):
        evt = ShipmentEvent(
            event_id=str(uuid.uuid4()),
            actor_id=actor.id,
            event_type=event_type,
            details=details,
            timestamp=datetime.utcnow().isoformat(),
            signature=actor.signer.sign(json.dumps({
                "tx_id": tx.tx_id,
                "event_type": event_type,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }, sort_keys=True, separators=(",", ":")))
        )
        tx.history.append(evt)
        tx.lifecycle_status = event_type
        # audit each lifecycle event
        self.audit.append({
            "event": "SHIPMENT_EVENT",
            "tx_id": tx.tx_id,
            "event_type": event_type,
            "actor": actor.id,
            "details": details,
            "event_id": evt.event_id
        }, signer=self.auditor.signer)
        return evt

    def dispatch(self, tx_id: str, dispatcher: Entity, est_departure: str):
        tx = self.pending_shipments.get(tx_id)
        if not tx:
            raise ValueError("shipment not found or already settled")
        return self._append_event(tx, dispatcher, "dispatched", {"est_departure": est_departure})

    def in_transit_update(self, tx_id: str, updater: Entity, location: str, eta: Optional[str]=None):
        tx = self.pending_shipments.get(tx_id)
        if not tx:
            raise ValueError("shipment not found or already settled")
        return self._append_event(tx, updater, "in_transit", {"location": location, "eta": eta})

    def customs_clearance(self, tx_id: str, customs_actor: Entity, clearance_doc: str):
        tx = self.pending_shipments.get(tx_id)
        if not tx:
            raise ValueError("shipment not found or already settled")
        return self._append_event(tx, customs_actor, "customs_cleared", {"clearance_doc": clearance_doc})

    def confirm_receipt(self, tx_id: str, receiver_actor: Entity, received_weight_kg: float, receiver_notes: Optional[str]=None):
        tx = self.pending_shipments.get(tx_id)
        if not tx:
            raise ValueError("shipment not found or already settled")
        evt = self._append_event(tx, receiver_actor, "receipt_confirmed", {"received_weight_kg": received_weight_kg, "notes": receiver_notes})
        # After receipt is confirmed, attempt settlement (create on-chain tx and propose)
        self.attempt_settlement(tx, receiver_actor)
        return evt

    def attempt_settlement(self, tx: SettlementTx, receipt_actor: Entity):
        # Create a settlement tx (payment leg) based on shipment; submitter must be present & already signed
        # tx already represents the payment record; ensure it has submitter signature (supplier/ops)
        if tx.tx_id not in self.pending_shipments:
            raise ValueError("shipment not pending")

        # Optionally attach a receipt signature into metadata for stronger proof
        tx.metadata["receipt_actor_id"] = receipt_actor.id
        tx.metadata["receipt_event_id"] = tx.history[-1].event_id if tx.history else None

        # If multisig required, approvals should already be present (procurement/compliance)
        proposer = self.validators[-1]  # simple proposer selection: last validator
        candidate_block = self.propose_block(proposer, [tx])
        committed, block = self.vote_block(candidate_block)
        self.audit.append({
            "event": "SETTLEMENT_ATTEMPT",
            "tx_id": tx.tx_id,
            "committed": committed,
            "block_index": block.index if committed else None
        }, signer=self.auditor.signer)
        return committed, block

    def reconcile(self, tx_id: str, reconciler: Entity, admitted_weight_kg: float):
        """
        Compare declared weight vs admitted weight and append reconciliation entry to audit log.
        Signed by reconciler actor.
        """
        tx = self._find_pending_or_committed(tx_id)
        if not tx:
            raise ValueError("tx not found")
        expected = tx.weight_kg
        variance = admitted_weight_kg - expected
        result = "match" if abs(variance) <= (0.01 * expected) else "mismatch"  # 1% tolerance
        # record reconciliation event
        evt = self._append_event(tx, reconciler, "reconciliation", {
            "declared_weight_kg": expected,
            "admitted_weight_kg": admitted_weight_kg,
            "variance": variance,
            "result": result
        })
        # audit reconciliation outcome
        self.audit.append({
            "event": "RECONCILIATION",
            "tx_id": tx.tx_id,
            "result": result,
            "variance": variance,
            "reconciler": reconciler.id,
            "recon_event_id": evt.event_id
        }, signer=self.auditor.signer)
        return evt, result

    def _find_pending_or_committed(self, tx_id: str) -> Optional[SettlementTx]:
        # search pending shipments
        if tx_id in self.pending_shipments:
            return self.pending_shipments[tx_id]
        # search in committed blocks
        for blk in self.blocks:
            for t in blk.txs:
                if t.tx_id == tx_id:
                    return t
        return None

# Global lookup to keep the demo simple
ENTITY_LOOKUP: Dict[str, Entity] = {}

# ---------------------------------------------------------------------
# IBC-style proof export (stub)
# ---------------------------------------------------------------------
def export_ibc_proof(ledger: Ledger, tx_id: str) -> Dict:
    for blk in ledger.blocks:
        for tx in blk.txs:
            if tx.tx_id == tx_id:
                header = {
                    "index": blk.index,
                    "prev_hash": blk.prev_hash,
                    "timestamp": blk.timestamp,
                    "tx_ids": [t.tx_id for t in blk.txs],
                    "proposer_id": blk.proposer_id,
                    "votes": blk.votes
                }
                return {
                    "proof_type": "demo_ibc_proof_bunge_v2",
                    "block_header": header,
                    "block_hash": blk.block_hash,
                    "tx_core": json.loads(tx.to_json()),
                    "audit_tail": "hash-chained in " + ledger.audit.path
                }
    raise ValueError("tx not found")

# ---------------------------------------------------------------------
# Demo / Main: create entities, run lifecycle, reconcile, settle
# ---------------------------------------------------------------------
def main():
    print("PQ Available:", PQ_AVAILABLE)
    ca = CertificateAuthority()

    # Auditor
    auditor = ca.enroll("Bunge Global Auditor", "auditor", quality_score=100, sustainability_score=100)

    # Validators
    v1 = ca.enroll("Bunge Validator - Americas", "validator")
    v2 = ca.enroll("Bunge Validator - EMEA", "validator")
    v3 = ca.enroll("Partner Validator - Logistics Consortium", "validator")
    validators = [v1, v2, v3]

    # Facilities
    bunge_hq = ca.enroll("Bunge HQ (St. Louis)", "bunge_hq", aml_score=1, quality_score=100, sustainability_score=95)
    port_spain = ca.enroll("Port Terminal - Barcelona", "port", aml_score=1, supply_cert=True, phyto=True, quality_score=95, sustainability_score=85)
    oilseed_plant_us = ca.enroll("Oilseed Processing Plant - US", "plant", aml_score=1, quality_score=92, sustainability_score=80)
    farmer_alpha = ca.enroll("Supplier Farm Alpha (Argentina)", "farmer", aml_score=1, supply_cert=True, phyto=True, quality_score=88, sustainability_score=70)
    logistics_inc = ca.enroll("Logistics Co. Global", "logistics", aml_score=1, supply_cert=True, quality_score=90, sustainability_score=80)
    customs_spain = ca.enroll("Spain Customs Authority (stub)", "customs", aml_score=1, kyc=True)

    for ent in [bunge_hq, port_spain, oilseed_plant_us, farmer_alpha, logistics_inc, customs_spain]:
        ENTITY_LOOKUP[ent.id] = ent

    # Policy
    policy = Policy(high_value_threshold=500_000, max_aml_score=3, approvers_required=2, min_quality_score=75, min_sustainability_score=60)
    ledger = Ledger(validators=validators, auditor=auditor, policy=policy)

    # Create shipment settlement tx (payment pending)
    tx = SettlementTx(
        tx_id=str(uuid.uuid4()),
        sender_facility_id=farmer_alpha.id,
        receiver_facility_id=oilseed_plant_us.id,
        amount_usd=600_000,  # high value -> requires multisig approvals
        commodity_type="soybean",
        batch_id="BATCH-ARG-2025-07-30-001",
        weight_kg=120000.0,
        created_at=datetime.utcnow().isoformat(),
        metadata={
            "purpose": "Bulk soybean shipment - payment conditional on receipt",
            "origin_country": "Argentina",
            "destination_country": "USA",
            "incoterm": "CFR",
            "bill_of_lading": "BOL-00012345"
        }
    )

    # Submitter (supplier ops) signs the core tx intent
    submitter = farmer_alpha
    tx.metadata["submitter_id"] = submitter.id
    tx.metadata["submitter_sig"] = submitter.signer.sign(tx.to_json())

    # Pre-approvals: procurement + compliance (Bunge HQ + port acceptance)
    approver1 = bunge_hq
    approver2 = port_spain
    tx.approvals[approver1.id] = approver1.signer.sign(tx.to_json())
    tx.approvals[approver2.id] = approver2.signer.sign(tx.to_json())

    # Register shipment in pending mempool
    ledger.register_shipment(tx)

    # Simulated lifecycle:
    # 1) Dispatch by logistics
    ledger.dispatch(tx.tx_id, dispatcher=logistics_inc, est_departure="2025-07-30T10:00:00Z")
    # 2) In-transit updates
    ledger.in_transit_update(tx.tx_id, updater=logistics_inc, location="Atlantic crossing, 1000km off coast", eta="2025-08-05T12:00:00Z")
    # 3) Customs clearance
    ledger.customs_clearance(tx.tx_id, customs_actor=customs_spain, clearance_doc="CUST-CLR-2025-08-04-778")
    # 4) Receipt confirmation at plant (received slightly under declared weight)
    ledger.confirm_receipt(tx.tx_id, receiver_actor=oilseed_plant_us, received_weight_kg=119800.0, receiver_notes="short by 200kg due to moisture loss")

    # At this point, settlement should have been attempted during confirm_receipt; check ledger state
    pending = tx.tx_id in ledger.pending_shipments
    print("\n--- Post-receipt status ---")
    print("Pending in mempool:", pending)
    # Attempt to export proof if committed
    committed_tx = None
    for blk in ledger.blocks:
        if any(t.tx_id == tx.tx_id for t in blk.txs):
            committed_tx = tx
            print("Found tx in block index:", blk.index)
            break

    # Reconciliation (compare declared vs admitted)
    evt, result = ledger.reconcile(tx.tx_id, reconciler=bunge_hq, admitted_weight_kg=119800.0)
    print("Reconciliation result:", result)
    print("Reconciliation event id:", evt.event_id)

    # If settlement committed, export IBC proof stub
    if committed_tx:
        proof = export_ibc_proof(ledger, tx.tx_id)
        print("\n=== IBC Proof (stub) ===")
        print(json.dumps(proof, indent=2))

    print("\nAudit log appended to:", ledger.audit.path)
    if os.path.exists(ledger.audit.path):
        with open(ledger.audit.path, "r") as f:
            last_line = list(f.readlines())[-1].strip()
            print("Last Audit Entry (truncated):", last_line[:200] + ("..." if len(last_line) > 200 else ""))

    # Show PQ availability & signer types for nodes (for debugging)
    print("\nSigner types:")
    for ent in [auditor, v1, v2, v3, bunge_hq, port_spain, oilseed_plant_us, farmer_alpha, logistics_inc, customs_spain]:
        print(f"- {ent.name}: {ent.signer.__class__.__name__}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Unhandled error:", e)
        traceback.print_exc()
        sys.exit(1)
