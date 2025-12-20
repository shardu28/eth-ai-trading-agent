# test_wallet_balance.py
from delta_client import DeltaClient

def main():
    client = DeltaClient()
    data = client.get_wallet_balance()

    if not data:
        raise RuntimeError("Empty response from wallet balance API")

    print("✅ Wallet balance API call successful")
    print("Keys returned:", data.keys())

if __name__ == "__main__":
    main()
