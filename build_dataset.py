import json
from datasets import load_dataset

def build_malicious():
    print("Loading malicious dataset (deepset/prompt-injections)...")
    try:
        ds = load_dataset("deepset/prompt-injections", split="train")
        malicious = [item['text'] for item in ds if item['label'] == 1]
        
        # We will just take up to 500 unique exemplars for the bank
        selected = list(set(malicious))[:500]
        
        with open("malicious_exemplars.json", "w") as f:
            json.dump(selected, f, indent=2)
        print(f"Saved {len(selected)} malicious exemplars.")
    except Exception as e:
        print(f"Error loading malicious dataset: {e}")

def build_benign():
    print("Loading benign dataset...")
    try:
        ds = load_dataset("deepset/prompt-injections", split="train")
        benign = [item['text'] for item in ds if item['label'] == 0]
        
        # We also want legitimate meta-questions. If deepset has none, 
        # we can just use the benign normal instructions.
        selected = list(set(benign))[:500]
        
        with open("benign_exemplars.json", "w") as f:
            json.dump(selected, f, indent=2)
        print(f"Saved {len(selected)} benign exemplars.")
    except Exception as e:
        print(f"Error loading benign dataset: {e}")

if __name__ == "__main__":
    build_malicious()
    build_benign()
