import json

with open("view/data.json", encoding="utf-8") as f:
    data = json.load(f)
    print(data)

print("over")
