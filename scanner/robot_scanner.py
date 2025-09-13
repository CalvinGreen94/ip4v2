#nmap_scanner.py

import nmap
import random
from collections import deque
from datetime import datetime
import time
import json
import flask 


# TASKS = [ 'delivery', 'inspection', 'painting', 'assembly']
SAFE_PORTS = {80, 443}
SUSPICIOUS_PORTS = [21, 22, 23, 3389, 8080, 1337]
ROBOT_TASKS = ["welding","inspection","delivery", "charging", "idle", "maintenance"]
SUSPICIOUS_PORTS = [23, 3389, 6667, 31337]

def get_random_ports():
    # Simulate ports accessed in each cycle
    return random.sample(range(20, 1024), 3) + random.sample(SUSPICIOUS_PORTS, 1)
def detect_intrusions(logs):
    print("\n🔍 Detecting anomalies...")
    for log in logs:
        alerts = []

        if any(port in SUSPICIOUS_PORTS for port in log["ports_accessed"]):
            alerts.append(f"Suspicious ports accessed: {log['ports_accessed']}")

        if len(log["task_history"]) >= 2:
            alerts.append(f"Unusual task switching: {log['task_history']}")

        if alerts:
            print(f"\n🚨 Intrusion detected at IP {log['ip']} during task '{log['current_task']}'")
            for alert in alerts:
                print("   ⚠️", alert)
from pqc.crypto_utils import generate_pqc_keys, sign_message, verify_signature

def run_nmap_scan(target='None', num_hosts=5, num_robots=5):
    scanner = nmap.PortScanner()
    all_results = []
    robot_logs = []
    time.sleep(5)

    for _ in range(num_hosts):
        current_target = target or "0.0.0.0"
        try:
            scanner.scan(hosts=current_target, arguments='-T4 -F --script vulners')
        except Exception as e:
            all_results.append({"error": str(e), "target": current_target})
            continue

        if not scanner.all_hosts():
            all_results.append({"warning": "No hosts found", "target": current_target})
            continue

        for host in scanner.all_hosts():
            host_data = {"host": host, "vulnerabilities": []}
            logs = []

            for _ in range(num_robots):
                ip = host
                num_task_switches = random.randint(0, 4)
                task_history = deque()
                ports_accessed = get_random_ports()
                current_task = random.choice(ROBOT_TASKS)
                previous_task = random.choice(ROBOT_TASKS)

                for _ in range(num_task_switches):
                    new_task = random.choice([t for t in ROBOT_TASKS if t != current_task])
                    task_history.append(new_task)
                    current_task = new_task

                accessed_ports = []
                is_intrusion = False
                reason = []
                if current_task != previous_task and previous_task != "idle":
                    is_intrusion = True
                    reason.append("Unusual Task Switch")

                if any(port in SUSPICIOUS_PORTS for port in ports_accessed):
                    is_intrusion = True
                    reason.append("Suspicious Port Access")

                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "ip": ip,
                    "task": current_task,
                    "previous_task": previous_task,
                    # "cpu": cpu,
                    # "network_traffic_MB": network_traffic,
                    "ports_accessed": ports_accessed,
                    "intrusion_detected": is_intrusion,
                    "reasons": reason if is_intrusion else []
                }

                robot_logs.append(log_entry)



                

                if is_intrusion:
                    print(f"🚨 ALERT @ {ip}: {reason}")
                    # log_to_blockchain(log_entry)


                for proto in scanner[host].all_protocols():
                    lport = scanner[host][proto].keys()
                    for port in lport:
                        port_data = scanner[host][proto][port]
                        vuln_output = port_data.get('script', {}).get('vulners', '')

                        # Add to vulnerabilities report
                        host_data["vulnerabilities"].append({
                            "port": port,
                            "protocol": proto,
                            "state": port_data.get('state'),
                            "name": port_data.get('name', ''),
                            "product": port_data.get('product', ''),
                            "version": port_data.get('version', ''),
                            "cve_output": vuln_output
                        })

                        accessed_ports.append(port)

                log = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "ip": ip,
                    "current_task": current_task,
                    "task_history": list(task_history),
                    "ports_accessed": accessed_ports
                }

                logs.append(log)
                robot_logs.append(log)

                # ✅ Show all robot activity
                print(f"\n🤖 Robot at {ip}")
                print(f"   Current Task: {current_task}")
                print(f"   Task History: {list(task_history)}")
                print(f"   Ports Accessed: {accessed_ports}")

            detect_intrusions(logs)
            
            all_results.append(host_data)

            # public_key, private_key = generate_pqc_keys()
            # message = f"ALERT: Port scan detected on {accessed_ports}, {ip}, {current_task}, {list(task_history)},"

            # signature = sign_message(private_key, message)
            # verified = verify_signature(public_key, message, signature)

            # return {
            #     "message": message,
            #     "signature": signature,
            #     "verified": verified
            # }
    # Save robot logs
    with open("robot_logs.json", "w") as f:
        json.dump(robot_logs, f, indent=4)

    print("\n📁 All robot logs saved to robot_logs.json")
    return all_results

# Example usage
# if __name__ == "__main__":
run_nmap_scan()
