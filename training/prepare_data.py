from datasets import load_dataset

def load_finance():
    ds = load_dataset("financial_phrasebank", "sentences_75agree")
    print(ds)
    print(ds['train'][0])
    return ds

if __name__ == "__main__":
    load_finance()
