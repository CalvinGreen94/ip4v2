#!/usr/bin/env python3
"""Bunge – Digital Commodity Settlement Network (Simulation) with Tanpin-Kanri Kanban logic.
Enhanced with comprehensive GPT-4o integration for AI-powered decision making.
This extends the original demo by adding:
 - InventoryItem and KanbanCard dataclasses
 - Automatic Kanban pull creation when on-hand <= reorder_point at receipt
 - Simple in-memory inventory bookkeeping (for demo only)
 - Comprehensive GPT-4o integration for predictions, anomaly detection, and insights
 - Helpers: compute_on_hand_for_sku, find_kanban_for_sku_and_consumer, create_kanban_pull
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib, hmac, json, uuid, os, sys, traceback
import time
from regenAGRI import simulate_regen_agri as sra
from openai import OpenAI

# Initialize OpenAI client with your API key
client = OpenAI(api_key="###")

# ---------------------------------------------------------------------
# Enhanced AI-powered utilities with GPT-4o
# ---------------------------------------------------------------------
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def ai_predict_inventory(facility_id: str, sku: str, historical_txns: List['SettlementTx']) -> Dict:
    """Enhanced inventory prediction with seasonal and market factors"""
    current_date = datetime.utcnow()
    season = "Q" + str((current_date.month - 1) // 3 + 1)
    
    prompt = f"""
    Analyze inventory trends for {sku} at facility {facility_id}.
    Current date: {current_date.isoformat()}
    Current season: {season}
    
    Historical transactions (last 30 days): 
    {json.dumps([t.to_json() for t in historical_txns[-10:]], indent=2)}
    
    Consider:
    - Seasonal demand patterns for {sku}
    - Supply chain disruptions
    - Market volatility
    - Lead times and buffer requirements
    
    Provide JSON output with:
    {{
      "predicted_on_hand_kg": <float>,
      "expected_shortage_next_7_days": <float>,
      "suggested_kanban_pull_qty": <float>,
      "confidence_level": <float 0-1>,
      "risk_factors": ["factor1", "factor2"],
      "recommended_actions": ["action1", "action2"]
    }}
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        result = json.loads(resp.choices[0].message.content)
        return result
    except Exception as e:
        print(f"AI prediction error: {e}")
        return {
            "predicted_on_hand_kg": 1000.0,
            "expected_shortage_next_7_days": 0.0,
            "suggested_kanban_pull_qty": 0.0,
            "confidence_level": 0.5,
            "risk_factors": ["AI system unavailable"],
            "recommended_actions": ["Use manual forecasting"]
        }

def ai_detect_anomalies(tx: 'SettlementTx', ledger: 'Ledger') -> Dict:
    """Enhanced anomaly detection with pattern recognition"""
    recent_txns = []
    for block in ledger.blocks[-5:]:  # Last 5 blocks
        recent_txns.extend([t.to_json() for t in block.txs])
    
    prompt = f"""
    Analyze this commodity transaction for anomalies and suspicious patterns:
    
    Current Transaction: {tx.to_json()}
    
    Recent Network Activity:
    {json.dumps(recent_txns, indent=2)}
    
    Check for:
    - Unusual price variations (>15% from market average)
    - Weight discrepancies
    - Suspicious routing patterns
    - Timing anomalies
    - Entity behavior patterns
    - Compliance red flags
    
    Output JSON format:
    {{
      "suspicious": <boolean>,
      "risk_score": <float 0-100>,
      "anomaly_type": "pricing|weight|routing|timing|entity|compliance|none",
      "reason": "detailed explanation",
      "recommended_action": "approve|review|reject|investigate",
      "additional_checks": ["check1", "check2"]
    }}
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"AI anomaly detection error: {e}")
        return {
            "suspicious": False,
            "risk_score": 0.0,
            "anomaly_type": "none",
            "reason": f"Unable to analyze - {str(e)}",
            "recommended_action": "review",
            "additional_checks": ["Manual review required"]
        }

def ai_optimize_kanban_parameters(sku: str, historical_data: List[Dict], current_kanban: 'KanbanCard') -> Dict:
    """AI-powered Kanban parameter optimization"""
    prompt = f"""
    Optimize Kanban parameters for SKU: {sku}
    
    Current Parameters:
    - Reorder Point: {current_kanban.reorder_point} kg
    - Target Level: {current_kanban.target_level} kg
    - Lot Size: {current_kanban.lot_size} kg
    - Lead Time: {current_kanban.lead_time_days} days
    
    Historical Usage Data:
    {json.dumps(historical_data, indent=2)}
    
    Consider:
    - Demand variability
    - Supply reliability
    - Storage costs
    - Seasonal patterns
    - Risk tolerance
    
    Output optimized parameters in JSON:
    {{
      "optimized_reorder_point": <float>,
      "optimized_target_level": <float>,
      "optimized_lot_size": <float>,
      "estimated_service_level": <float>,
      "cost_impact": <float>,
      "justification": "explanation of changes"
    }}
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"AI Kanban optimization error: {e}")
        return {
            "optimized_reorder_point": current_kanban.reorder_point,
            "optimized_target_level": current_kanban.target_level,
            "optimized_lot_size": current_kanban.lot_size,
            "estimated_service_level": 0.9,
            "cost_impact": 0.0,
            "justification": f"Optimization unavailable: {str(e)}"
        }

def ai_generate_dashboard(ledger: 'Ledger') -> str:
    """Enhanced dashboard with AI insights"""
    pending_count = len(ledger.pending_shipments)
    recent_blocks = len([b for b in ledger.blocks if b.index > 1])  # Exclude genesis
    total_inventory = sum(item.quantity for item in ledger.inventory.values())
    active_kanbans = len([k for k in ledger.kanbans.values() if k.status == "open"])
    
    prompt = f"""
    Generate an executive dashboard summary for Bunge's Digital Settlement Network:
    
    Current Status:
    - Pending Shipments: {pending_count}
    - Settled Blocks: {recent_blocks}
    - Total Inventory: {total_inventory} kg across facilities
    - Active Kanban Cards: {active_kanbans}
    
    Recent Activity Summary:
    {json.dumps([{
        'block': b.index,
        'tx_count': len(b.txs),
        'timestamp': b.timestamp
    } for b in ledger.blocks[-3:]], indent=2)}
    
    Generate a professional summary including:
    - Network health status
    - Key performance indicators
    - Risk alerts
    - Operational recommendations
    - Market insights
    
    Format as a clear, executive-level report.
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1200
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"""
        BUNGE DIGITAL SETTLEMENT NETWORK - DASHBOARD
        ============================================
        Status: OPERATIONAL (AI Analytics Unavailable)
        
        Current Metrics:
        • Pending Shipments: {pending_count}
        • Settled Blocks: {recent_blocks}
        • Total Inventory: {total_inventory:,.0f} kg
        • Active Kanban Cards: {active_kanbans}
        
        Note: Advanced AI insights temporarily unavailable ({str(e)})
        Contact system administrator for manual analysis.
        """

def ai_generate_event_notes(tx: 'SettlementTx', event_type: str) -> str:
    """Enhanced event note generation with context awareness"""
    prompt = f"""
    Generate professional event notes for a {event_type} event:
    
    Transaction Details:
    - ID: {tx.tx_id}
    - Commodity: {tx.commodity_type}
    - Weight: {tx.weight_kg:,.0f} kg
    - Value: ${tx.amount_usd:,}
    - Route: {tx.sender_facility_id} → {tx.receiver_facility_id}
    - Batch: {tx.batch_id}
    
    Event Type: {event_type}
    
    Include relevant:
    - Quality considerations for {tx.commodity_type}
    - Compliance checkpoints
    - Risk factors
    - Next steps
    
    Keep professional, concise (2-3 sentences max).
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Event {event_type} completed for {tx.commodity_type} shipment {tx.batch_id}. Standard protocols applied."

def ai_explain_variance(tx: 'SettlementTx', admitted_weight_kg: float) -> str:
    """Enhanced variance explanation with industry knowledge"""
    variance_pct = ((admitted_weight_kg - tx.weight_kg) / tx.weight_kg) * 100
    
    prompt = f"""
    Explain the weight variance for this commodity shipment:
    
    Transaction: {tx.commodity_type} shipment
    Declared Weight: {tx.weight_kg:,.0f} kg
    Admitted Weight: {admitted_weight_kg:,.0f} kg
    Variance: {variance_pct:+.2f}%
    
    For {tx.commodity_type} shipments, consider typical causes:
    - Moisture loss/gain during transport
    - Handling losses
    - Scale calibration differences  
    - Natural settling/compaction
    - Temperature/humidity effects
    
    Provide a clear, professional explanation suitable for stakeholders.
    Include whether this variance is within acceptable industry standards.
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        return resp.choices[0].message.content
    except Exception as e:
        if abs(variance_pct) <= 1.0:
            return f"Weight variance of {variance_pct:+.2f}% is within acceptable tolerance for {tx.commodity_type} shipments."
        else:
            return f"Weight variance of {variance_pct:+.2f}% requires investigation. Potential causes include moisture changes or handling losses."

def ai_assess_supply_chain_risk(ledger: 'Ledger', entity_id: str) -> Dict:
    """AI-powered supply chain risk assessment"""
    entity_txns = []
    for block in ledger.blocks:
        for tx in block.txs:
            if tx.sender_facility_id == entity_id or tx.receiver_facility_id == entity_id:
                entity_txns.append(tx.to_json())
    
    prompt = f"""
    Assess supply chain risk for entity: {entity_id}
    
    Transaction History:
    {json.dumps(entity_txns[-10:], indent=2)}
    
    Evaluate:
    - Transaction volume trends
    - Delivery performance
    - Quality consistency  
    - Geographic concentration risk
    - Counterparty diversity
    - Seasonal patterns
    
    Output JSON format:
    {{
      "overall_risk_score": <float 0-100>,
      "risk_category": "low|medium|high|critical",
      "key_risks": ["risk1", "risk2"],
      "mitigation_recommendations": ["action1", "action2"],
      "performance_trend": "improving|stable|declining",
      "diversification_score": <float 0-100>
    }}
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {
            "overall_risk_score": 50.0,
            "risk_category": "medium",
            "key_risks": ["Limited analysis available"],
            "mitigation_recommendations": ["Implement manual risk monitoring"],
            "performance_trend": "stable",
            "diversification_score": 50.0
        }

def ai_generate_compliance_report(tx: 'SettlementTx', compliance_data: Dict) -> str:
    """AI-generated compliance and regulatory report"""
    prompt = f"""
    Generate a compliance report for commodity transaction:
    
    Transaction: {tx.to_json()}
    
    Compliance Data:
    {json.dumps(compliance_data, indent=2)}
    
    Cover:
    - Regulatory compliance status
    - Documentation completeness
    - Risk assessment
    - Required follow-up actions
    - Audit trail summary
    
    Format as a professional compliance report.
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Compliance Report for {tx.tx_id}: Standard regulatory requirements met. Documentation complete. No exceptions noted."

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
except Exception:
    PQ_AVAILABLE = False

if PQ_AVAILABLE:
    class PQSigner(BaseSigner):
        def __init__(self, name: str, seed: Optional[bytes] = None):
            super().__init__(name)
            self.public_key, self.secret_key = dilithium2.generate_keypair()

        def sign(self, message: str) -> str:
            sig = dilithium2.sign(message.encode(), self.secret_key)
            return sig.hex()

        def verify(self, message: str, signature: str) -> bool:
            try:
                sig_bytes = bytes.fromhex(signature)
                msg_out = dilithium2.open(sig_bytes, self.public_key)
                return msg_out.decode() == message
            except Exception:
                return False
else:
    class HMACSigner(BaseSigner):
        def __init__(self, name: str, secret: Optional[str] = None):
            super().__init__(name)
            self.secret = (secret or sha256(uuid.uuid4().bytes)[:32]).encode()

        def sign(self, message: str) -> str:
            return hmac.new(self.secret, message.encode(), hashlib.sha256).hexdigest()

        def verify(self, message: str, signature: str) -> bool:
            expected = self.sign(message)
            return hmac.compare_digest(expected, signature)

def create_signer(name: str, secret: Optional[str] = None):
    if PQ_AVAILABLE:
        return PQSigner(name)
    else:
        return HMACSigner(name, secret)

# ---------------------------------------------------------------------
# Enhanced Entity with AI risk scoring
# ---------------------------------------------------------------------
@dataclass
class Entity:
    id: str
    name: str
    role: str                # "validator" | "bunge_hq" | "plant" | "port" | "farmer" | "logistics" | "auditor" | "customs"
    signer: BaseSigner
    kyc_verified: bool = True
    aml_risk_score: int = 1
    supply_cert_verified: bool = True
    quality_score: int = 90
    phytosanitary_cert: bool = True
    sustainability_score: int = 80
    ai_risk_profile: Optional[Dict] = None  # AI-generated risk assessment

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
# Enhanced Inventory + Kanban with AI optimization
# ---------------------------------------------------------------------
@dataclass
class InventoryItem:
    item_id: str            # UUID or SKU-batch identifier
    sku: str
    batch_id: str
    owner_facility_id: str
    quantity: float
    unit: str = "kg"
    quality_score: int = 90
    sustainability_score: int = 80
    location: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    ai_quality_assessment: Optional[Dict] = None  # AI quality predictions

@dataclass
class KanbanCard:
    id: str
    sku: str
    provider_id: str
    consumer_id: str
    reorder_point: float
    target_level: float
    lot_size: float
    lead_time_days: int
    status: str = "open"
    last_pull_ts: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    ai_optimization_data: Optional[Dict] = None  # AI-suggested parameters

# ---------------------------------------------------------------------
# Enhanced Transactions, Blocks, Policy with AI integration
# ---------------------------------------------------------------------
@dataclass
class ShipmentEvent:
    event_id: str
    actor_id: str
    event_type: str  # dispatch | in_transit | customs_cleared | receipt_confirmed | reconciliation | kanban_pull
    details: Dict
    timestamp: str
    signature: str
    ai_notes: Optional[str] = None  # AI-generated event notes

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
    lifecycle_status: str = "created"  # created, dispatched, in_transit, customs_cleared, receipt_confirmed, settled
    history: List[ShipmentEvent] = field(default_factory=list)
    ai_anomaly_score: Optional[float] = None  # AI risk assessment
    ai_recommendations: List[str] = field(default_factory=list)  # AI suggestions

    def to_json(self) -> str:
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
    ai_summary: Optional[str] = None  # AI-generated block summary

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
                 min_sustainability_score: int = 60,
                 ai_risk_threshold: float = 0.7):  # AI anomaly threshold
        self.high_value_threshold = high_value_threshold
        self.max_aml_score = max_aml_score
        self.approvers_required = approvers_required
        self.min_quality_score = min_quality_score
        self.min_sustainability_score = min_sustainability_score
        self.ai_risk_threshold = ai_risk_threshold

    def requires_multisig(self, amount: int) -> bool:
        return amount >= self.high_value_threshold

    def requires_ai_review(self, ai_risk_score: float) -> bool:
        return ai_risk_score >= self.ai_risk_threshold

# ---------------------------------------------------------------------
# Enhanced Ledger with comprehensive AI integration
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
        self.pending_shipments: Dict[str, SettlementTx] = {}

        # Enhanced in-memory registries
        self.inventory: Dict[str, InventoryItem] = {}  # item_id -> InventoryItem
        self.kanbans: Dict[str, KanbanCard] = {}       # kanban_id -> KanbanCard
        self.ai_insights: Dict[str, Dict] = {}         # tx_id -> AI insights

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
        
        # Generate AI summary for the block
        try:
            summary_prompt = f"""
            Summarize this block of {len(txs)} transactions:
            {json.dumps([{'tx_id': t.tx_id, 'commodity': t.commodity_type, 'value': t.amount_usd, 'weight': t.weight_kg} for t in txs], indent=2)}
            
            Provide a brief professional summary of the block contents and any notable patterns.
            """
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            block.ai_summary = resp.choices[0].message.content.strip()
        except Exception:
            block.ai_summary = f"Block {block.index} contains {len(txs)} transactions"
        
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
                "votes": list(votes.keys()),
                "ai_summary": block.ai_summary
            }, signer=self.auditor.signer)
            
            for t in block.txs:
                t.lifecycle_status = "settled"
                if t.tx_id in self.pending_shipments:
                    del self.pending_shipments[t.tx_id]
            return True, block
        else:
            return False, block

    def _validate_txs(self, txs: List[SettlementTx]) -> bool:
        for tx in txs:
            # Enhanced validation with AI anomaly detection
            anomaly_result = ai_detect_anomalies(tx, self)
            tx.ai_anomaly_score = anomaly_result.get("risk_score", 0.0) / 100.0
            
            if anomaly_result.get("suspicious", False):
                tx.ai_recommendations.append(f"AI flagged: {anomaly_result.get('reason', 'Unknown')}")
                if self.policy.requires_ai_review(tx.ai_anomaly_score):
                    return False
            
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
    # Enhanced Inventory & Kanban helpers with AI optimization
    # --------------------------
    def register_inventory(self, item: InventoryItem):
        self.inventory[item.item_id] = item
        
        # AI quality assessment for new inventory
        try:
            quality_prompt = f"""
            Assess quality outlook for commodity inventory:
            - SKU: {item.sku}
            - Quantity: {item.quantity} {item.unit}
            - Current Quality Score: {item.quality_score}
            
            IMPORTANT: Respond ONLY with valid JSON. No additional text.
            
            {{
              "predicted_shelf_life_days": 120,
              "quality_trend": "stable",
              "storage_recommendations": ["maintain_temperature_control", "monitor_moisture"],
              "risk_factors": ["seasonal_humidity_changes"]
            }}
            """
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a commodity quality assessment AI. Respond only with valid JSON objects."},
                    {"role": "user", "content": quality_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            response_text = resp.choices[0].message.content.strip()
            
            # Clean up response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]
            
            item.ai_quality_assessment = json.loads(response_text)
        except Exception as e:
            # Provide commodity-specific defaults
            shelf_life = 180 if item.sku == "wheat" else 120 if item.sku == "soybean" else 90
            item.ai_quality_assessment = {
                "predicted_shelf_life_days": shelf_life,
                "quality_trend": "stable",
                "storage_recommendations": ["standard_commodity_storage", "regular_quality_monitoring"],
                "risk_factors": [f"{item.sku}_specific_risks", "environmental_factors"]
            }
        
        self.audit.append({
            "event": "INVENTORY_REGISTERED",
            "item_id": item.item_id,
            "sku": item.sku,
            "qty": item.quantity,
            "owner": item.owner_facility_id,
            "ai_quality_assessment": item.ai_quality_assessment
        }, signer=self.auditor.signer)

    def compute_on_hand_for_sku(self, facility_id: str, sku: str) -> float:
        total = 0.0
        for it in self.inventory.values():
            if it.owner_facility_id == facility_id and it.sku == sku:
                total += it.quantity
        return total

    def find_kanban_for_sku_and_consumer(self, sku: str, consumer_id: str) -> Optional[KanbanCard]:
        for k in self.kanbans.values():
            if k.sku == sku and k.consumer_id == consumer_id and k.status == "open":
                return k
        return None

    def create_kanban(self, sku: str, provider_id: str, consumer_id: str, reorder_point: float, target_level: float, lot_size: float, lead_time_days: int) -> KanbanCard:
        kid = str(uuid.uuid4())
        k = KanbanCard(
            id=kid, 
            sku=sku, 
            provider_id=provider_id, 
            consumer_id=consumer_id, 
            reorder_point=reorder_point, 
            target_level=target_level, 
            lot_size=lot_size, 
            lead_time_days=lead_time_days
        )
        
        # AI optimization of Kanban parameters
        historical_data = self._get_historical_usage_data(sku, consumer_id)
        k.ai_optimization_data = ai_optimize_kanban_parameters(sku, historical_data, k)
        
        self.kanbans[kid] = k
        self.audit.append({
            "event": "KANBAN_CREATED",
            "kanban_id": kid,
            "sku": sku,
            "provider": provider_id,
            "consumer": consumer_id,
            "reorder_point": reorder_point,
            "ai_optimization": k.ai_optimization_data
        }, signer=self.auditor.signer)
        return k

    def _get_historical_usage_data(self, sku: str, facility_id: str) -> List[Dict]:
        """Extract historical usage patterns for AI analysis"""
        usage_data = []
        for block in self.blocks[-10:]:  # Last 10 blocks
            for tx in block.txs:
                if tx.receiver_facility_id == facility_id and tx.commodity_type == sku:
                    usage_data.append({
                        "date": tx.created_at,
                        "quantity": tx.weight_kg,
                        "source": tx.sender_facility_id
                    })
        return usage_data

    def create_kanban_pull(self, kanban: KanbanCard, qty: float, triggered_by: Entity):
        # AI-enhanced pull quantity optimization
        historical_txns = [tx for block in self.blocks for tx in block.txs 
                          if tx.commodity_type == kanban.sku]
        
        ai_prediction = ai_predict_inventory(kanban.consumer_id, kanban.sku, historical_txns)
        optimized_qty = ai_prediction.get("suggested_kanban_pull_qty", qty)
        
        if optimized_qty > 0 and abs(optimized_qty - qty) / qty > 0.1:  # >10% difference
            qty = optimized_qty  # Use AI recommendation
            
        order_tx = SettlementTx(
            tx_id=str(uuid.uuid4()),
            sender_facility_id=kanban.provider_id,
            receiver_facility_id=kanban.consumer_id,
            amount_usd=0,
            commodity_type=kanban.sku,
            batch_id=f"KANBAN-{kanban.id}-{datetime.utcnow().isoformat()}",
            weight_kg=qty,
            created_at=datetime.utcnow().isoformat(),
            metadata={
                "purpose": "kanban_pull",
                "kanban_id": kanban.id, 
                "triggered_by": triggered_by.id,
                "ai_optimized_qty": optimized_qty,
                "ai_prediction": ai_prediction
            }
        )
        
        submitter = self._find_entity(kanban.provider_id)
        if submitter:
            order_tx.metadata["submitter_id"] = submitter.id
            order_tx.metadata["submitter_sig"] = submitter.signer.sign(order_tx.to_json())
            
        self.register_shipment(order_tx)
        self.audit.append({
            "event": "KANBAN_PULL_CREATED",
            "kanban_id": kanban.id,
            "original_qty": qty,
            "ai_optimized_qty": optimized_qty,
            "triggered_by": triggered_by.id,
            "order_tx": order_tx.tx_id,
            "ai_prediction": ai_prediction
        }, signer=self.auditor.signer)
        
        kanban.last_pull_ts = datetime.utcnow().isoformat()
        return order_tx

    # --------------------------
    # Enhanced Shipment lifecycle with AI insights
    # --------------------------
    def register_shipment(self, tx: SettlementTx):
        # AI anomaly detection during registration
        anomaly_result = ai_detect_anomalies(tx, self)
        tx.ai_anomaly_score = anomaly_result.get("risk_score", 0.0) / 100.0
        
        if anomaly_result.get("suspicious", False):
            tx.ai_recommendations.append(f"Registration alert: {anomaly_result.get('reason', 'Unknown')}")
        
        self.pending_shipments[tx.tx_id] = tx
        self.ai_insights[tx.tx_id] = anomaly_result
        
        self.audit.append({
            "event": "SHIPMENT_REGISTERED",
            "tx_id": tx.tx_id,
            "batch_id": tx.batch_id,
            "sender": tx.sender_facility_id,
            "receiver": tx.receiver_facility_id,
            "amount_usd": tx.amount_usd,
            "ai_anomaly_score": tx.ai_anomaly_score,
            "ai_recommendations": tx.ai_recommendations
        }, signer=self.auditor.signer)

    def _append_event(self, tx: SettlementTx, actor: Entity, event_type: str, details: Dict):
        # Enhanced event creation with AI insights
        ai_notes = ai_generate_event_notes(tx, event_type)
        
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
            }, sort_keys=True, separators=(",", ":"))),
            ai_notes=ai_notes
        )
        
        details['ai_notes'] = ai_notes
        tx.history.append(evt)
        tx.lifecycle_status = event_type
        
        # Update AI insights for this transaction
        if tx.tx_id in self.ai_insights:
            self.ai_insights[tx.tx_id][f"{event_type}_notes"] = ai_notes
        
        self.audit.append({
            "event": "SHIPMENT_EVENT",
            "tx_id": tx.tx_id,
            "event_type": event_type,
            "actor": actor.id,
            "details": details,
            "event_id": evt.event_id,
            "ai_notes": ai_notes
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
        
        # AI analysis of weight variance
        if abs(received_weight_kg - tx.weight_kg) > 0.01 * tx.weight_kg:  # >1% variance
            variance_explanation = ai_explain_variance(tx, received_weight_kg)
            if receiver_notes:
                receiver_notes += f" AI Analysis: {variance_explanation}"
            else:
                receiver_notes = f"AI Analysis: {variance_explanation}"
        
        evt = self._append_event(tx, receiver_actor, "receipt_confirmed", {
            "received_weight_kg": received_weight_kg, 
            "notes": receiver_notes
        })
        
        # Attach receipt metadata
        tx.metadata["receipt_actor_id"] = receiver_actor.id
        tx.metadata["receipt_event_id"] = evt.event_id

        # Enhanced Tanpin/Kanban logic with AI predictions
        new_item = InventoryItem(
            item_id=str(uuid.uuid4()), 
            sku=tx.commodity_type, 
            batch_id=tx.batch_id, 
            owner_facility_id=receiver_actor.id, 
            quantity=received_weight_kg
        )
        self.register_inventory(new_item)

        on_hand = self.compute_on_hand_for_sku(receiver_actor.id, tx.commodity_type)
        kanban = self.find_kanban_for_sku_and_consumer(tx.commodity_type, receiver_actor.id)
        
        if kanban and on_hand <= kanban.reorder_point:
            # AI-enhanced Kanban pull decision
            historical_txns = [t for block in self.blocks for t in block.txs 
                              if t.commodity_type == tx.commodity_type]
            ai_prediction = ai_predict_inventory(receiver_actor.id, tx.commodity_type, historical_txns)
            
            if ai_prediction.get("expected_shortage_next_7_days", 0) > 0:
                self.create_kanban_pull(kanban, qty=kanban.lot_size, triggered_by=receiver_actor)

        # Attempt settlement
        self.attempt_settlement(tx, receiver_actor)
        return evt

    def attempt_settlement(self, tx: SettlementTx, receipt_actor: Entity):
        if tx.tx_id not in self.pending_shipments:
            raise ValueError("shipment not pending")
            
        tx.metadata["receipt_actor_id"] = receipt_actor.id
        tx.metadata["receipt_event_id"] = tx.history[-1].event_id if tx.history else None
        
        proposer = self.validators[-1]  # Simple proposer selection
        candidate_block = self.propose_block(proposer, [tx])
        committed, block = self.vote_block(candidate_block)
        
        self.audit.append({
            "event": "SETTLEMENT_ATTEMPT",
            "tx_id": tx.tx_id,
            "committed": committed,
            "block_index": block.index if committed else None,
            "ai_block_summary": block.ai_summary
        }, signer=self.auditor.signer)
        return committed, block

    def reconcile(self, tx_id: str, reconciler: Entity, admitted_weight_kg: float):
        tx = self._find_pending_or_committed(tx_id)
        if not tx:
            raise ValueError("tx not found")
            
        expected = tx.weight_kg
        variance = admitted_weight_kg - expected
        result = "match" if abs(variance) <= (0.01 * expected) else "mismatch"
        
        # Enhanced reconciliation with AI variance explanation
        ai_explanation = ai_explain_variance(tx, admitted_weight_kg)
        
        evt = self._append_event(tx, reconciler, "reconciliation", {
            "declared_weight_kg": expected,
            "admitted_weight_kg": admitted_weight_kg,
            "variance": variance,
            "result": result,
            "ai_explanation": ai_explanation
        })
        
        self.audit.append({
            "event": "RECONCILIATION",
            "tx_id": tx.tx_id,
            "result": result,
            "variance": variance,
            "reconciler": reconciler.id,
            "recon_event_id": evt.event_id,
            "ai_explanation": ai_explanation
        }, signer=self.auditor.signer)
        return evt, result

    def _find_pending_or_committed(self, tx_id: str) -> Optional[SettlementTx]:
        if tx_id in self.pending_shipments:
            return self.pending_shipments[tx_id]
        for blk in self.blocks:
            for t in blk.txs:
                if t.tx_id == tx_id:
                    return t
        return None

    def generate_ai_dashboard(self) -> str:
        """Generate comprehensive AI-powered dashboard"""
        return ai_generate_dashboard(self)

    def run_ai_supply_chain_analysis(self, entity_id: str) -> Dict:
        """Run comprehensive AI supply chain risk analysis"""
        return ai_assess_supply_chain_risk(self, entity_id)

    def optimize_all_kanbans(self):
        """AI optimization of all active Kanban cards"""
        optimized_count = 0
        for kanban in self.kanbans.values():
            if kanban.status == "open":
                historical_data = self._get_historical_usage_data(kanban.sku, kanban.consumer_id)
                optimization = ai_optimize_kanban_parameters(kanban.sku, historical_data, kanban)
                
                # Apply optimizations if improvement is significant
                if optimization.get("cost_impact", 0) < -0.05:  # >5% cost reduction
                    kanban.reorder_point = optimization.get("optimized_reorder_point", kanban.reorder_point)
                    kanban.target_level = optimization.get("optimized_target_level", kanban.target_level)
                    kanban.lot_size = optimization.get("optimized_lot_size", kanban.lot_size)
                    kanban.ai_optimization_data = optimization
                    optimized_count += 1
                    
                    self.audit.append({
                        "event": "KANBAN_AI_OPTIMIZATION",
                        "kanban_id": kanban.id,
                        "optimization_data": optimization
                    }, signer=self.auditor.signer)
        
        return optimized_count

# Global lookup to keep the demo simple
ENTITY_LOOKUP: Dict[str, Entity] = {}

# ---------------------------------------------------------------------
# Enhanced IBC-style proof export with AI insights
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
                    "proof_type": "ai_enhanced_ibc_proof_bunge_v3",
                    "block_header": header,
                    "block_hash": blk.block_hash,
                    "block_ai_summary": blk.ai_summary,
                    "tx_core": json.loads(tx.to_json()),
                    "tx_ai_insights": ledger.ai_insights.get(tx_id, {}),
                    "audit_tail": "hash-chained in " + ledger.audit.path,
                    "sra": sra()
                }
    raise ValueError("tx not found")

# ---------------------------------------------------------------------
# Enhanced Demo with comprehensive AI integration
# ---------------------------------------------------------------------
def main():
    print("=== BUNGE DIGITAL SETTLEMENT NETWORK with AI (GPT-4o) ===")
    print("PQ Available:", PQ_AVAILABLE)
    
    ca = CertificateAuthority()

    # Create entities
    auditor = ca.enroll("G&G FOODS Inc", "auditor", quality_score=100, sustainability_score=100)
    
    validators = [
        ca.enroll("Bunge Validator - Americas", "validator"),
        ca.enroll("Bunge Validator - EMEA", "validator"),
        ca.enroll("Partner Validator - Logistics Consortium", "validator")
    ]

    # Facilities
    bunge_hq = ca.enroll("Bunge HQ (St. Louis)", "bunge_hq", aml_score=1, quality_score=100, sustainability_score=95)
    port_spain = ca.enroll("Port Terminal - Barcelona", "port", aml_score=1, supply_cert=True, phyto=True, quality_score=95, sustainability_score=85)
    oilseed_plant_us = ca.enroll("Oilseed Processing Plant - US", "plant", aml_score=1, quality_score=92, sustainability_score=80)
    farmer_alpha = ca.enroll("Supplier Farm Alpha (Argentina)", "farmer", aml_score=1, supply_cert=True, phyto=True, quality_score=88, sustainability_score=70)
    logistics_inc = ca.enroll("Logistics Co. Global", "logistics", aml_score=1, supply_cert=True, quality_score=90, sustainability_score=80)
    customs_spain = ca.enroll("Spain Customs Authority", "customs", aml_score=1, kyc=True)

    for ent in [bunge_hq, port_spain, oilseed_plant_us, farmer_alpha, logistics_inc, customs_spain]:
        ENTITY_LOOKUP[ent.id] = ent

    # Enhanced policy with AI thresholds
    policy = Policy(
        high_value_threshold=500_000, 
        max_aml_score=3, 
        approvers_required=2, 
        min_quality_score=75, 
        min_sustainability_score=60,
        ai_risk_threshold=0.7
    )
    ledger = Ledger(validators=validators, auditor=auditor, policy=policy)

    print("\n--- AI-Enhanced Kanban Setup ---")
    time.sleep(3)
    # Create AI-optimized Kanban card
    kanban = ledger.create_kanban(
        sku="soybean", 
        provider_id=farmer_alpha.id, 
        consumer_id=oilseed_plant_us.id, 
        reorder_point=10000.0, 
        target_level=50000.0, 
        lot_size=20000.0, 
        lead_time_days=10
    )
    print(f"Kanban created with AI optimization: {kanban.ai_optimization_data}")

    # Preload inventory near reorder point
    initial_item = InventoryItem(
        item_id=str(uuid.uuid4()), 
        sku="soybean", 
        batch_id="ONHAND-BATCH-001", 
        owner_facility_id=oilseed_plant_us.id, 
        quantity=9500.0
    )
    ledger.register_inventory(initial_item)
    print(f"Initial inventory AI assessment: {initial_item.ai_quality_assessment}")

    print("\n--- AI-Enhanced Transaction Processing ---")
    time.sleep(3)
    # Create high-value shipment transaction
    tx = SettlementTx(
        tx_id=str(uuid.uuid4()),
        sender_facility_id=farmer_alpha.id,
        receiver_facility_id=oilseed_plant_us.id,
        amount_usd=600_000,
        commodity_type="soybean",
        batch_id="BATCH-ARG-2025-07-30-001",
        weight_kg=120000.0,
        created_at=datetime.utcnow().isoformat(),
        metadata={
            "purpose": "AI-enhanced bulk soybean shipment",
            "origin_country": "Argentina",
            "destination_country": "USA",
            "incoterm": "CFR",
            "bill_of_lading": "BOL-00012345"
        }
    )

    # Set up transaction signatures and approvals
    submitter = farmer_alpha
    tx.metadata["submitter_id"] = submitter.id
    tx.metadata["submitter_sig"] = submitter.signer.sign(tx.to_json())

    # Multi-signature approvals for high-value transaction
    tx.approvals[bunge_hq.id] = bunge_hq.signer.sign(tx.to_json())
    tx.approvals[port_spain.id] = port_spain.signer.sign(tx.to_json())

    # Register with AI anomaly detection
    ledger.register_shipment(tx)
    print(f"Transaction registered. AI anomaly score: {tx.ai_anomaly_score:.3f}")
    print(f"AI recommendations: {tx.ai_recommendations}")

    print("\n--- AI-Enhanced Lifecycle Management ---")
    time.sleep(3)
    # Execute lifecycle with AI insights
    ledger.dispatch(tx.tx_id, dispatcher=logistics_inc, est_departure="2025-07-30T10:00:00Z")
    ledger.in_transit_update(tx.tx_id, updater=logistics_inc, location="Atlantic crossing", eta="2025-08-05T12:00:00Z")
    ledger.customs_clearance(tx.tx_id, customs_actor=customs_spain, clearance_doc="CUST-CLR-2025-08-04-778")

    # Receipt confirmation with AI variance analysis
    ledger.confirm_receipt(tx.tx_id, receiver_actor=oilseed_plant_us, received_weight_kg=119800.0, receiver_notes="Minor moisture loss during transit")

    print("\n--- AI-Driven Kanban Analysis ---")
    time.sleep(3)
    on_hand_after = ledger.compute_on_hand_for_sku(oilseed_plant_us.id, "soybean")
    print(f"On-hand inventory after receipt: {on_hand_after:,.0f} kg")
    
    kanban_pulls = [t for t in ledger.pending_shipments.values() if t.metadata.get("purpose") == "kanban_pull"]
    print(f"AI-triggered Kanban pulls: {len(kanban_pulls)}")
    for p in kanban_pulls:
        ai_data = p.metadata.get("ai_prediction", {})
        print(f"- Pull order {p.tx_id}: {p.weight_kg:,.0f} kg (AI confidence: {ai_data.get('confidence_level', 0):.2f})")

    print("\n--- AI-Enhanced Reconciliation ---")
    time.sleep(3)
    evt, result = ledger.reconcile(tx.tx_id, reconciler=bunge_hq, admitted_weight_kg=119800.0)
    print(f"Reconciliation result: {result}")
    print(f"AI variance explanation: {evt.details.get('ai_explanation', 'N/A')}")

    print("\n--- AI Dashboard Generation ---")
    time.sleep(3)
    dashboard = ledger.generate_ai_dashboard()
    print(dashboard)

    print("\n--- AI Supply Chain Risk Analysis ---")
    time.sleep(3)
    risk_analysis = ledger.run_ai_supply_chain_analysis(farmer_alpha.id)
    print(f"Risk assessment for {farmer_alpha.name}:")
    print(f"- Overall risk score: {risk_analysis.get('overall_risk_score', 0):.1f}/100")
    print(f"- Risk category: {risk_analysis.get('risk_category', 'unknown')}")
    print(f"- Key risks: {risk_analysis.get('key_risks', [])}")

    print("\n--- Kanban AI Optimization ---")
    time.sleep(3)
    optimized_count = ledger.optimize_all_kanbans()
    print(f"AI optimized {optimized_count} Kanban cards")

    # Export enhanced IBC proof
    committed_tx = None
    for blk in ledger.blocks:
        if any(t.tx_id == tx.tx_id for t in blk.txs):
            committed_tx = tx
            print(f"Transaction settled in block {blk.index}")
            break

    if committed_tx:
        proof = export_ibc_proof(ledger, tx.tx_id)
        print("\n--- AI-Enhanced IBC Proof ---")
        print(f"Proof type: {proof['proof_type']}")
        print(f"Block AI summary: {proof['block_ai_summary']}")
        print(f"Transaction AI insights available: {len(proof['tx_ai_insights'])} entries")

    print(f"\nAudit log: {ledger.audit.path}")
    print(f"AI-enhanced entries logged with comprehensive insights")

    # Show entity signer types
    print("\n--- System Configuration ---")
    print("Signer types:")
    for ent in [auditor] + validators + [bunge_hq, port_spain, oilseed_plant_us, farmer_alpha, logistics_inc, customs_spain]:
        print(f"- {ent.name}: {ent.signer.__class__.__name__}")
    
    print(f"AI Integration: GPT-4o model active")
    print(f"Enhanced features: Anomaly detection, Inventory prediction, Kanban optimization")

if __name__ == "__main__":
    try:
        while True:
            time.sleep(10)  # Increased interval for AI processing
            main()
            # sra()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()
        sys.exit(1)