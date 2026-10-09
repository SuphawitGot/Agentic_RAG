from datasets import load_dataset

ds = load_dataset("mtybilly/apex-r1-real-world-documents")

print(ds)                       # Splits and row counts
print(ds["train"].column_names)  # Available columns

for index, row in enumerate(ds["train"].select(range(15))):
    print(index, repr(row["text"]))