"""Container readiness, including actual default-category serving validation."""
import json
import urllib.request


def main():
    with urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=20) as response:
        if json.load(response)['status'] != 'ready':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
