"""RagSale backend package."""

def main():
    from .ingest import main as ingest
    ingest()
