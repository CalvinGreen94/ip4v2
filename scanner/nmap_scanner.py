#scanner/nmap_scanner.py
import nmap
import random
def run_nmap_scan(target=None, num_hosts=5):
    scanner = nmap.PortScanner()
    all_results = []

    for _ in range(num_hosts):
        current_target = target or f"192.168.1.{random.randint(2, 254)}"

        try:
            # Add --script vulners to identify CVEs
            scanner.scan(hosts=current_target, arguments='-T4 -F --script vulners')
        except Exception as e:
            all_results.append({"error": str(e), "target": current_target})
            continue

        if not scanner.all_hosts():
            all_results.append({"warning": "No hosts found", "target": current_target})
            continue

        for host in scanner.all_hosts():
            host_data = {"host": host, "vulnerabilities": []}
            for proto in scanner[host].all_protocols():
                lport = scanner[host][proto].keys()
                for port in lport:
                    port_data = scanner[host][proto][port]
                    vuln_output = port_data.get('script', {}).get('vulners', '')

                    host_data["vulnerabilities"].append({
                        "port": port,
                        "protocol": proto,
                        "state": port_data.get('state'),
                        "name": port_data.get('name', ''),
                        "product": port_data.get('product', ''),
                        "version": port_data.get('version', ''),
                        "cve_output": vuln_output
                    })

            all_results.append(host_data)

    return all_results

# Example usage
results = run_nmap_scan(num_hosts=3)
for r in results:
    print(r)
