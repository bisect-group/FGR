# FGR - The Functional Group Representation Module.

## Installation

Clone the repository and install in editable mode (recommended for development):

```sh
pip install git+https://github.com/bisectgroup/FGR.git
```

## Usage

After installation, you can use the FGR encoder in your Python code or notebooks:

```python
from FGR.encoder import FGREncoder

embedder = FGREncoder(
    device="cpu"  # or "cuda" if using GPU
)

smiles_list = ["CCN", "CCF"]
emb = embedder.get_embedding(smiles_list, method="FGR")
print(emb.shape)
```

- `method` can be `"FG"`, `"MFG"`, or `"FGR"` depending on the embedding you want.
- Make sure the required files (`FGR_model.ckpt`, `fg.parquet`, tokenizer) are available at the specified paths.