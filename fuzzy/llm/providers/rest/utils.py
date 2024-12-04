import json
from typing import Any, Dict


def parse_http_request(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    method, path, _ = lines[0].split()
    headers = {}
    for line in lines[1:]:
        if line == '\n':
            break
        key, value = line.strip().split(': ')
        headers[key] = value
    
    body = json.loads(lines[-1])
    
    return {
        "method": method,
        "path": path,
        "headers": headers,
        "body": body
    }
