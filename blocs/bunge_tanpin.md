# Bunge Digital Settlement Network (Simulation with AI + Kanban)

## Overview
This project simulates a **digital commodity settlement network** for Bunge using blockchain-inspired logic, integrated with **AI (GPT-4o)** for enhanced decision-making.  
The system combines:
- **Inventory control with Tanpin-Kanri Kanban logic**  
- **AI-enhanced analysis** for anomaly detection, reconciliation, and optimization  
- **Blockchain-style settlement ledger** with HMAC-based signing for participants  
- **Executive dashboards** summarizing network health and risks  

The simulation is designed for demonstration and research, not production trading.  

---

## Features

### 1. AI-Enhanced Kanban Setup
- AI attempts to optimize **reorder point, target levels, and lot sizes**.  
- Provides **inventory assessments** (shelf life prediction, quality trends, storage recommendations).  
- Creates **Kanban cards** when on-hand stock falls below the reorder point.  
- Example AI output:
  ```json
  {
    "optimized_reorder_point": 10000.0,
    "optimized_target_level": 50000.0,
    "optimized_lot_size": 20000.0,
    "estimated_service_level": 0.9
  }
