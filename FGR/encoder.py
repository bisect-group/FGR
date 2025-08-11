import sys
import torch
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from typing import List, Literal, Optional
from huggingface_hub import hf_hub_download
from rdkit.Chem.rdmolfiles import MolFromSmarts

from FGR.src.data.components.utils import (
    smiles2vector_fg,
    smiles2vector_mfg,
    standardize_smiles,
)
from FGR.src.models.fgr_module import FGRPretrainLitModule

if 'src' not in sys.modules:
    import FGR.src
    sys.modules['src'] = FGR.src
    sys.modules['src.models'] = FGR.src.models
    sys.modules['src.models.fgr_module'] = FGR.src.models.fgr_module

import warnings
warnings.filterwarnings("ignore", module="lightning.pytorch.utilities.parsing")

class FGREncoder:
    def __init__(
        self,
        ckpt_path: Optional[str] = hf_hub_download(repo_id="bisectgroup/FGR", filename="FGR_model.ckpt"),
        fg_parquet_path: Optional[str] = hf_hub_download(repo_id="bisectgroup/FGR", filename="fg.parquet"),
        tokenizer_path: Optional[str] = hf_hub_download(repo_id="bisectgroup/FGR", filename="BPE_pubchem_500.json"),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Load functional groups
        fgroups = pd.read_parquet(fg_parquet_path)["SMARTS"].tolist()
        self.fgroups_list = [MolFromSmarts(x) for x in fgroups]
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.device = device

        # Load model if checkpoint path is given
        self.model = None
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.model = FGRPretrainLitModule(**checkpoint['hyper_parameters'])
            self.model.load_state_dict(checkpoint['state_dict'])
            #self.model = FGRPretrainLitModule.load_from_checkpoint(ckpt_path, map_location=device, weights_only=True)
            self.model.eval()
            self.model.to(device)

    def get_representation(
        self,
        smiles: List[str],
        method: str,
    ) -> np.ndarray:
        smiles = [standardize_smiles(smi) for smi in smiles]  # Standardize smiles
        if method == "FG":
            x = np.stack([smiles2vector_fg(x, self.fgroups_list) for x in smiles])
        elif method == "MFG":
            x = np.stack([smiles2vector_mfg(x, self.tokenizer) for x in smiles])
        elif method == "FGR":
            f_g = np.stack([smiles2vector_fg(x, self.fgroups_list) for x in smiles])
            mfg = np.stack([smiles2vector_mfg(x, self.tokenizer) for x in smiles])
            x = np.concatenate((f_g, mfg), axis=1)  # Concatenate both vectors
        else:
            raise ValueError("Method not supported")
        return x

    def get_embedding(
        self,
        smiles: List[str],
        method: Literal["FG", "MFG", "FGR"] = "FGR"
    ) -> np.ndarray:
        x_tensor = torch.tensor(self.get_representation(smiles, method, ), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self.model(x_tensor)[0].detach().cpu().numpy()
        return z

# Example usage:
# from FGR.src.embedder import FGREmbedder
# embedder = FGREmbedder(ckpt_path="./FGR_model.ckpt")
# emb = embedder.get_embedding(["CCN", "CCF"], method="FGR")