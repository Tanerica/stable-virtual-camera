# read_prompts.py
prompts = []
with open("caption100.txt", "r", encoding="utf-8") as f:
    # strip removes newline characters (\n)
    prompts = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(prompts)} prompts:")
for i, p in enumerate(prompts[:5]):  # show first 5 to check
    print(f"{i+1}: {p}")
