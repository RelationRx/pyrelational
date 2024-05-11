import os
import pickle
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MolStandardize
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from pyrelational.datasets.download_utils import (
    create_directory_if_not_exists,
    download_file,
    fetch_data,
)

logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


class DrugCombDataset(Dataset[Tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Pytorch dataset for DrugComb drug combination data."""

    ALLOWED_SYNERGY = ["zip", "loewe", "hsa", "bliss"]
    ALLOWED_STUDIES = {
        "GDSC1",
        "CTRPV2",
        "CCLE",
        "GCSI",
        "NCATS_ES(FAKI/AURKI)",
        "NCATS_DIPG",
        "GRAY",
        "FIMM",
        "ALMANAC",
        "FLOBAK",
        "FORCINA",
        "YOHE",
        "NCATS_HL",
        "WILSON",
        "ASTRAZENECA",
        "SCHMIDT",
        "UHNBREAST",
        "ONEIL",
        "NCATS_MDR_CS",
        "CLOUD",
        "PHELAN",
        "MATHEWS",
        "NCATS_ATL",
        "BEATAML",
        "NCATS_SARS-COV-2DPI",
        "BOBROWSKI",
        "DYALL",
        "MOTT",
        "FRIEDMAN",
        "FALLAHI-SICHANI",
        "FRIEDMAN2",
        "MILLER",
    }

    DRUGCOMB_SUMMARY_URL = "https://drugcomb.fimm.fi/jing/summary_v_1_5.csv"
    DRUGS_API_URL = "https://api.drugcomb.org/drugs"
    CELL_LINE_API_URL = "https://api.drugcomb.org/cell_lines"
    CCLE_EXPRESSION_URL = "https://ndownloader.figshare.com/files/34989919"

    def __init__(
        self,
        root: str = ".",
        studies: Optional[list[str]] = None,
        synergy_score: str = "bliss",
        n_splits: int = 1,
        test_size: Union[float, int] = 0.2,
        random_seed: int = 0,
        cell_lines_embed_dim: Optional[int] = 512,
    ):
        """Create instance of DrugCombDataset.

        :param root: path to folder where the data should be cached, defaults to "."
        :param studies: list of studies to restrict the data to, defaults to None (i.e. ["ALMANAC"])
        :param synergy_score: which synergy score to use, defaults to "bliss"
        :param n_splits: number of splits, defaults to 1
        :param test_size: size of the test set, either in proportion (float) or in absolute size (int), defaults to 0.2
        :param random_seed: random seed for reproducibility, defaults to 0
        :param cell_lines_embed_dim: dimension of cell lines embedding, defaults to 128
            - if != None, we use PCA to reduce the dimension to stipulated dimension
            - if == None, we use the original dimension
        """
        if studies is None:
            studies = ["ALMANAC"]
        self.validate_inputs(studies, synergy_score)

        self.root = os.path.join(root, type(self).__name__)
        self.seed = random_seed
        self.studies = studies
        self.synergy_score = f"synergy_{synergy_score}"
        self.cell_lines_embed_dim = cell_lines_embed_dim

        self.setup_paths(studies, synergy_score, cell_lines_embed_dim)
        create_directory_if_not_exists(self.root)

        if not self.load_cache():
            self.setup_dataset()

        self._create_splits(n_splits=n_splits, random_seed=random_seed, test_size=test_size)

    def validate_inputs(self, studies: list[str], synergy_score: str) -> None:
        """Check user input against allowed values"""
        assert (
            synergy_score in self.ALLOWED_SYNERGY
        ), f"{synergy_score} not supported, pick a synergy score in {self.ALLOWED_SYNERGY}"
        assert all(study in self.ALLOWED_STUDIES for study in studies), (
            f"Some studies are not allowed: {set(studies) - self.ALLOWED_STUDIES}."
            f"Make sure they are all in {self.ALLOWED_STUDIES}."
        )

    def setup_paths(self, studies: list[str], synergy_score: str, cell_line_dim: Optional[int]) -> None:
        """Set up data_directory path."""
        dataset_name = f"{'_'.join(studies)}-{synergy_score}"
        if cell_line_dim is not None:
            dataset_name += f"-cell_dim:{cell_line_dim}"
        self.data_directory = os.path.join(self.root, dataset_name)

    def setup_dataset(self) -> None:
        """Setup data structures."""
        drug_to_id = self.set_drug_features()
        cell_to_id = self.set_cell_line_features()
        self.download_and_process_drugcomb_data(drug_to_id, cell_to_id)
        self.cache_data()

    def set_drug_features(self) -> Dict[str, int]:
        """Download and sets drug fingerprint."""
        drug_data = fetch_data(self.DRUGS_API_URL)
        drug_to_smiles = {
            drug["dname"]: drug["smiles"].split(";")[0]
            for drug in drug_data
            if (drug["dname"] != "NULL") and (drug["smiles"] != "NULL")
        }
        drug_to_id = {drug["dname"]: drug["id"] for drug in drug_data if drug["dname"] != "NULL"}

        self.drug_id_to_fingerprint = {
            drug_to_id[name]: torch.from_numpy(self.parse_smiles_to_fingerprint(smiles)).float()
            for name, smiles in tqdm(drug_to_smiles.items(), desc="Converting drug SMILES to fingerprints")
            if self.parse_smiles_to_fingerprint(smiles) is not None
        }
        return {k: v for k, v in drug_to_id.items() if v in self.drug_id_to_fingerprint}

    @staticmethod
    def parse_smiles_to_fingerprint(smiles: str) -> Optional[NDArray[np.int_]]:
        """Convert drug smiles string to morgan fingerprints."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            fragment = MolStandardize.rdMolStandardize.ChargeParent(mol)
            cmol = MolStandardize.rdMolStandardize.TautomerEnumerator().Canonicalize(fragment)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(cmol, radius=2, nBits=1024)
            array = np.zeros((2048,), dtype=int)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            return array
        except Exception as e:
            print(f"Failed to parse {smiles}: {e}")
            return None

    def set_cell_line_features(self) -> Dict[str, int]:
        """Download and set cell line features."""
        cell_line_data = fetch_data(self.CELL_LINE_API_URL)
        expression_data = pd.read_csv(download_file(self.CCLE_EXPRESSION_URL, self.root), index_col=0)
        expression_data.index = expression_data.index.map({d["depmap_id"]: d["id"] for d in cell_line_data})
        expression_data = expression_data[~expression_data.index.isna()]
        if self.cell_lines_embed_dim is not None:
            expression_data = self.reduce_dim_with_pca(expression_data, self.cell_lines_embed_dim)
        self.cell_id_to_expression = {
            int(idx): torch.from_numpy(data.values).float() for idx, data in expression_data.iterrows()
        }
        return {cell["name"]: cell["id"] for cell in cell_line_data if cell["id"] in self.cell_id_to_expression}

    @staticmethod
    def reduce_dim_with_pca(data: pd.DataFrame, dim: int) -> pd.DataFrame:
        """Reduce data dimension using PCA.

        :param data: dataframe containing data
        :param dim: dimension to reduce to
        """
        pca = PCA(n_components=dim)
        transformed_data = pca.fit_transform(data)
        return pd.DataFrame(data=transformed_data, index=data.index)

    def download_and_process_drugcomb_data(self, drug_to_id: Dict[str, int], cell_to_id: Dict[str, int]) -> None:
        """Download and process synergy data from drugcomb."""
        summary = pd.read_csv(download_file(self.DRUGCOMB_SUMMARY_URL, self.root))
        summary = summary[
            (summary.study_name.isin(self.studies))
            & (summary.drug_row.isin(drug_to_id))
            & (summary.drug_col.isin(drug_to_id))
            & (summary.cell_line_name.isin(cell_to_id))
        ]

        drug_row = summary.drug_row.apply(drug_to_id.get).values
        drug_col = summary.drug_col.apply(drug_to_id.get).values
        cell_id = summary.cell_line_name.apply(cell_to_id.get).values
        self.indices = np.stack((drug_row, drug_col, cell_id), axis=1).astype(int)
        self.y = torch.from_numpy(summary[self.synergy_score].values).float()
        self.drug_id_to_fingerprint = {
            k: v for k, v in self.drug_id_to_fingerprint.items() if (k in drug_row) or (k in drug_col)
        }
        self.cell_id_to_expression = {k: v for k, v in self.cell_id_to_expression.items() if (k in cell_id)}

    def cache_data(self) -> None:
        """Cache data for easier reload."""
        create_directory_if_not_exists(self.data_directory)

        with open(os.path.join(self.data_directory, "indices.pkl"), "wb") as f:
            pickle.dump(self.indices, f)

        with open(os.path.join(self.data_directory, "synergy_scores.pkl"), "wb") as f:
            pickle.dump(self.y, f)

        with open(os.path.join(self.data_directory, "drug_fingerprints.pkl"), "wb") as f:
            pickle.dump(self.drug_id_to_fingerprint, f)

        with open(os.path.join(self.data_directory, "cell_line_expression.pkl"), "wb") as f:
            pickle.dump(self.cell_id_to_expression, f)

    def load_cache(self) -> bool:
        """Load cached data."""
        try:
            with open(os.path.join(self.data_directory, "indices.pkl"), "rb") as f:
                self.indices = pickle.load(f)
            with open(os.path.join(self.data_directory, "synergy_scores.pkl"), "rb") as f:
                self.y = pickle.load(f)
            with open(os.path.join(self.data_directory, "drug_fingerprints.pkl"), "rb") as f:
                self.drug_id_to_fingerprint = pickle.load(f)
            with open(os.path.join(self.data_directory, "cell_line_expression.pkl"), "rb") as f:
                self.cell_id_to_expression = pickle.load(f)
            return True
        except FileNotFoundError:
            return False

    def _create_splits(self, n_splits: int = 1, test_size: Union[float, int] = 0.2, random_seed: int = 0) -> None:
        """Create splits for the data."""
        splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
        self.data_splits = list(splitter.split(self.y))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return item associate to index.

        Each item is a combinations with [drug1_idx, drug2_idx, cellline_idx, synergy_score]
        returns fingerprints for drugs 1&2, cell-line expression, synergy_score.
        """
        drug_x, drug_y, cell_id = self.indices[index]
        return (
            self.drug_id_to_fingerprint[drug_x],
            self.drug_id_to_fingerprint[drug_y],
            self.cell_id_to_expression[cell_id],
            self.y[index],
        )

    def __len__(self) -> int:
        return len(self.indices)
