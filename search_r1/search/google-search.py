import requests

queries = "what is spatial proteomics"

res = requests.post(
    "http://localhost:8000/retrieve",
    json= {
        "queries": [queries]
    }
)


print("status:", res.status_code)
print("text:", res.text)

try:
    print("json:", res.json())
except Exception:
    print("response is not JSON")