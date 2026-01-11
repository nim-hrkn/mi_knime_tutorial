
from torch_geometric.datasets import QM9

# QM9 データセットをdata/qm9ディレクトリにダウンロードする。

import pandas as pd

def load_qm9_dataframe(root="data/qm9"):

    # -----------------------------
    # 1. Load QM9 dataset (PyG)
    # -----------------------------
    dataset = QM9(root=root)
    print("Total molecules:", len(dataset))

    IDX_HOMO = 2
    IDX_LUMO = 3
    IDX_GAP = 4
    
    # PyG の dataset をそのまま pandas Series にする
    s = pd.Series(dataset)
    
    # HOMO/LUMO/GAP を 1 サンプルずつ抽出する関数
    def extract_y(data):
        y = data.y[0]  # shape [19]
        return pd.Series({
            "SMILES": data.smiles,
            "HOMO":   y[IDX_HOMO].item(),
            "LUMO":   y[IDX_LUMO].item(),
            "GAP":    y[IDX_GAP].item(),
        })
    
    # apply で一気に DataFrame に展開
    df = s.apply(extract_y)
    return df    

df = load_qm9_dataframe()

print(df.head())
