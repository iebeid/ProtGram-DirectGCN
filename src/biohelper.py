
import os
import sys
import csv
import re
import logging
import time
import random
from collections import Counter
from typing import Union, Tuple, Optional, List, Dict, Iterator, Any, Callable

import requests
import pandas as pd
import dask.dataframe as dd
import pyarrow as pa  # Import PyArrow for schema definition
from Bio import SeqIO
from tqdm import tqdm
import mysql.connector

from helper import Algorithms

# --- Configuration ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "adminadmin",
    "database": "pdi"
}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def _compute_char_edges_for_sequence(sequence_text: str,
                                     strip_algo: Callable[[str], str] = Algorithms.strip_non_alphanumeric,
                                     encode_algo: Callable[[List[str]], List[int]] = Algorithms.encode_characters
                                     ) -> List[Dict[str, Union[str, float]]]:
    if not sequence_text or not isinstance(sequence_text, str): return []
    main_body = strip_algo(sequence_text)
    if not main_body: return []
    final_character_walk: List[str] = list(main_body)
    final_character_walk.append(" ")
    encoded_walk: List[int] = encode_algo(final_character_walk)
    if len(encoded_walk) < 2: return []
    edges = [f"{encoded_walk[i]}|{encoded_walk[i + 1]}" for i in range(len(encoded_walk) - 1)]
    if not edges: return []
    edge_counter = Counter(edges)
    character_edges: List[Dict[str, Union[str, float]]] = []
    for edge_str, count in edge_counter.items():
        parts = edge_str.split("|")
        if len(parts) == 2:
            try:
                character_edges.append({'from_node': parts[0], 'to_node': parts[1], 'weight': float(count)})
            except ValueError:
                logger.warning(f"Could not parse edge components {parts[0]}, {parts[1]} for edge '{edge_str}'")
        else:
            logger.warning(f"Malformed edge string '{edge_str}' found.")
    return character_edges


class ProteinOps:  # Unchanged
    @staticmethod
    def load_sequences_with_dask(filepath: str, sequence_column_name: str = "Sequence",
                                 blocksize: str = '128MB', dtype_for_sequence_col: str = 'str'
                                 ) -> Optional[dd.DataFrame]:
        if not os.path.exists(filepath): logger.error(f"File not found at {filepath}"); return None
        logger.info(f"Initiating Dask to process file: {filepath} with blocksize: {blocksize}")
        try:
            ddf = dd.read_csv(filepath, sep='\t', usecols=[sequence_column_name],
                              blocksize=blocksize, dtype={sequence_column_name: dtype_for_sequence_col})
            logger.info(f"Successfully created Dask DataFrame for column '{sequence_column_name}'. Partitions: {ddf.npartitions}")
            return ddf
        except ValueError as ve:
            if f"'{sequence_column_name}'" in str(ve) and "is not in columns" in str(ve):
                logger.error(f"Column '{sequence_column_name}' not found in {filepath}.")
            else:
                logger.error(f"ValueError during Dask loading: {ve}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Dask loading: {e}"); return None

    @staticmethod
    def compute_sequence_lengths_dask(sequences_series: dd.Series) -> Optional[dd.Series]:
        if not isinstance(sequences_series, dd.Series): logger.error("Input must be a Dask Series."); return None
        return sequences_series.str.len()

    @staticmethod
    def compute_sequence_edges_partitioned(sequences: List[str]
                                           ) -> List[List[Dict[str, Union[str, float]]]]:
        all_sequence_edges: List[List[Dict[str, Union[str, float]]]] = []
        for seq_text in sequences:
            all_sequence_edges.append(_compute_char_edges_for_sequence(seq_text))
        return all_sequence_edges


class PDBOps:  # Unchanged
    @staticmethod
    def get_protein_id(protein_name: str) -> Optional[str]:
        if not protein_name or not isinstance(protein_name, str): return None
        return protein_name.split('_')[0]

    @staticmethod
    def get_protein_name_from_id(protein_id: str) -> Optional[str]:
        if not protein_id or not isinstance(protein_id, str): return None
        return f"{protein_id}_protein"

    @staticmethod
    def download_pdb_structure(pdb_id: str, output_dir: str = ".", file_format: str = "pdb",
                               timeout: int = 30
                               ) -> Tuple[Optional[str], Optional[str]]:
        if file_format not in ['pdb', 'cif']: logger.error(f"Invalid file format '{file_format}'."); return None, None
        download_url = f"https://files.rcsb.org/download/{pdb_id.lower()}.{file_format}"
        logger.info(f"Attempting to download: {download_url}")
        try:
            response = requests.get(download_url, timeout=timeout);
            response.raise_for_status()
            pdb_data = response.text
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{pdb_id.lower()}.{file_format}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pdb_data)
            logger.info(f"Structure saved to: {output_path}")
            return output_path, pdb_data
        except requests.exceptions.HTTPError as he:
            if he.response.status_code == 404:
                logger.error(f"PDB ID '{pdb_id}' format '{file_format}' not found (404).")
            else:
                logger.error(f"HTTP error for {pdb_id}: {he.response.status_code} - {he.response.text[:200]}")
            return None, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {pdb_id}: {e}"); return None, None


class DaskOps:
    @staticmethod
    def save_dask_dataframe_to_parquet(dask_df: dd.DataFrame, directory_path: str,
                                       engine: str = 'pyarrow', overwrite: bool = True,
                                       compression: Optional[str] = 'gzip',
                                       schema: Optional[pa.Schema] = None) -> bool:  # Added schema parameter
        if dask_df is None: logger.error("Dask DataFrame is None. Nothing to save."); return False
        if not isinstance(dask_df, dd.DataFrame): logger.error("Input is not a Dask DataFrame."); return False
        if dask_df.npartitions == 0:
            logger.warning("Attempting to save an empty Dask DataFrame (0 partitions). Skipping save to Parquet.")
            return False

        logger.info(f"Saving Dask DataFrame to Parquet at: {directory_path} using compression: {compression}, engine: {engine}")
        try:
            # Dask's to_parquet with overwrite=True should handle directory.
            # No need for manual os.makedirs here if relying on Dask.
            kwargs_for_to_parquet = {
                "write_index": False,
                "engine": engine,
                "overwrite": overwrite,
                "compression": compression
            }
            if engine == 'pyarrow' and schema is not None:
                kwargs_for_to_parquet['schema'] = schema
                logger.info("Using explicit PyArrow schema for saving.")
            elif schema is not None and engine != 'pyarrow':
                logger.warning(f"Schema provided but engine is '{engine}', not 'pyarrow'. Schema might not be used by engine.")

            dask_df.to_parquet(directory_path, **kwargs_for_to_parquet)
            logger.info(f"Successfully saved Dask DataFrame to {directory_path}")
            return True
        except Exception as e:
            # The error message already contains "Failed to convert partition to expected pyarrow schema"
            logger.error(f"Error saving Dask DataFrame to Parquet: {e}", exc_info=True)
            return False

    @staticmethod
    def load_dask_dataframe_from_parquet(directory_path: str, columns: Optional[List[str]] = None,
                                         engine: str = 'pyarrow'
                                         ) -> Optional[dd.DataFrame]:
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            logger.error(f"Parquet directory not found or not a dir: {directory_path}");
            return None
        logger.info(f"Loading Dask DataFrame from Parquet at: {directory_path}")
        try:
            ddf = dd.read_parquet(directory_path, columns=columns, engine=engine)
            logger.info(f"Successfully loaded Dask DataFrame from {directory_path}")
            if ddf is not None: logger.info(f"  Columns: {ddf.columns.tolist()}, Partitions: {ddf.npartitions}")
            return ddf
        except Exception as e:
            if "smaller than the minimum file footer" in str(e) or "Invalid Parquet file size" in str(e):
                logger.error(f"Error loading Parquet: File in {directory_path} likely empty/corrupt. {e}")
            else:
                logger.error(f"Error loading Dask DataFrame from Parquet: {e}", exc_info=True)
            return None


class DatabaseOps:  # Unchanged
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        try:
            conn_test = mysql.connector.connect(host=self.db_config["host"], user=self.db_config["user"], password=self.db_config["password"])
            cursor_test = conn_test.cursor()
            cursor_test.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}")
            logger.info(f"Database '{self.db_config['database']}' ensured to exist.")
            cursor_test.close();
            conn_test.close()
        except mysql.connector.Error as err:
            logger.error(f"Failed to connect/ensure database '{self.db_config['database']}': {err}")

    def _get_connection(self) -> mysql.connector.MySQLConnection:
        try:
            conn = mysql.connector.connect(**self.db_config); return conn  # type: ignore
        except mysql.connector.Error as err:
            logger.error(f"Database connection error: {err}"); raise

    def execute_ddl(self, ddl_statement: str) -> bool:
        conn = None;
        cursor = None
        try:
            conn = self._get_connection();
            cursor = conn.cursor()
            for stmt in ddl_statement.split(';'):
                if stmt.strip(): cursor.execute(stmt.strip())
            conn.commit();
            logger.info(f"Successfully executed DDL: {ddl_statement.splitlines()[0][:100]}...")
            return True
        except mysql.connector.Error as err:
            logger.error(f"Error executing DDL '{ddl_statement.splitlines()[0][:100]}...': {err}")
            if conn: conn.rollback(); return False
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()

    def insert_data_batch(self, table_name: str, columns: List[str],
                          values_batch: List[Tuple[Any, ...]]) -> bool:
        if not values_batch: return True
        conn = None;
        cursor = None
        try:
            conn = self._get_connection();
            cursor = conn.cursor()
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(f"`{col}`" for col in columns)
            sql = f"INSERT INTO `{table_name}` ({column_names}) VALUES ({placeholders})"
            cursor.executemany(sql, values_batch);
            conn.commit()
            logger.info(f"{cursor.rowcount} record(s) inserted into {table_name}.")
            return True
        except mysql.connector.Error as err:
            logger.error(f"DB batch insert error into {table_name}: {err}")
            if conn: conn.rollback(); return False
        except Exception as e:
            logger.error(f"Unexpected error during batch insert into {table_name}: {e}")
            if conn: conn.rollback(); return False
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()


class FileParsingOps:  # Unchanged
    @staticmethod
    def parse_fasta_efficiently(file_path: str) -> Iterator[Dict[str, Any]]:
        try:
            with open(file_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    yield {'identifier': record.id, 'description': record.description,
                           'length': len(record.seq), 'sequence': str(record.seq)}
        except FileNotFoundError:
            logger.error(f"FASTA file not found: '{file_path}'")
        except Exception as e:
            logger.error(f"Error parsing FASTA file '{file_path}': {e}")

    @staticmethod
    def parse_fasta_to_dict(file_path: str, default_source: str = "UNKNOWN") -> Dict[str, Dict[str, Any]]:
        sequences: Dict[str, Dict[str, Any]] = {};
        current_id: Optional[str] = None
        current_seq_lines: List[str] = [];
        current_description: str = ""
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line: continue
                    if line.startswith('>'):
                        if current_id:
                            sequences[current_id] = {'source': default_source, 'description': current_description,
                                                     'length': len("".join(current_seq_lines)), 'sequence': "".join(current_seq_lines)}
                        header_parts = line[1:].split(maxsplit=1)
                        current_id = header_parts[0]
                        current_description = header_parts[1] if len(header_parts) > 1 else ""
                        current_seq_lines = []
                    elif current_id:
                        current_seq_lines.append(line)
                if current_id:
                    sequences[current_id] = {'source': default_source, 'description': current_description,
                                             'length': len("".join(current_seq_lines)), 'sequence': "".join(current_seq_lines)}
        except FileNotFoundError:
            logger.error(f"FASTA file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error parsing FASTA file to dict '{file_path}': {e}")
        return sequences

    @staticmethod
    def parse_generic_tsv(filepath: str, delimiter: str = '\t', skip_header_lines: int = 0) -> Iterator[List[str]]:
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as fh:
                reader = csv.reader(fh, delimiter=delimiter)
                for _ in range(skip_header_lines):
                    try:
                        next(reader)
                    except StopIteration:
                        logger.warning(f"File '{filepath}' has < {skip_header_lines} lines."); return
                for row in reader: yield row
        except FileNotFoundError:
            logger.error(f"File not found: '{filepath}'")
        except Exception as e:
            logger.error(f"Error parsing TSV file '{filepath}': {e}")

    @staticmethod
    def parse_uniref_fasta_to_tsv_rows(input_fasta_path: str) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(input_fasta_path): logger.error(f"Input FASTA missing: '{input_fasta_path}'"); return
        logger.info(f"Starting parsing of UniRef FASTA: '{input_fasta_path}'")
        try:
            for record in FileParsingOps.parse_fasta_efficiently(input_fasta_path):
                desc = record['description'];
                org, tax, rep = "N/A", "N/A", "N/A"
                if m := re.search(r"OS=(.*?)(?:\s(?:OX=|GN=|PE=|SV=)|$)", desc): org = m.group(1).strip()
                if m := re.search(r"OX=(\d+)", desc): tax = m.group(1).strip()
                if m := re.search(r"RepID=([\S]+)", desc): rep = m.group(1).strip()
                yield {'UniRef_ID': record['identifier'], 'Organism': org,
                       'Sequence_Length': record['length'], 'Tax_ID': tax,
                       'Rep_ID': rep, 'Sequence': record['sequence']}
        except Exception as e:
            logger.error(f"An error occurred during UniRef FASTA parsing: {e}")


class BioWorkflowOps:  # Unchanged
    @staticmethod
    def dask_partition_processor(partition_df: pd.DataFrame) -> pd.DataFrame:
        if 'Sequence' not in partition_df.columns:
            logger.warning("Partition missing 'Sequence' column.")
            for col, dtype in [('Sequence_Length', 'int64'), ('Sequence_Edges', 'object')]:
                if col not in partition_df: partition_df[col] = pd.Series(dtype=dtype)
            return partition_df

        if partition_df.empty:
            for col, dtype in [('Sequence_Length', 'int64'), ('Sequence_Edges', 'object')]:
                if col not in partition_df: partition_df[col] = pd.Series(dtype=dtype)
            return partition_df

        sequences_series = partition_df['Sequence'].fillna("").astype(str)
        partition_df['Sequence_Length'] = sequences_series.str.len().astype('int64')
        sequences_list: List[str] = sequences_series.tolist()
        partition_df['Sequence_Edges'] = ProteinOps.compute_sequence_edges_partitioned(sequences_list)
        return partition_df


def tests():
    logger.info("Starting example workflow...")
    run_suffix = f"_{int(time.time())}"

    # base_data_dir = os.path.join("G:", "My Drive", "Knowledge", "Research", "TWU", "Projects", "Link Prediction in Protein Interaction Networks via Structural Sequence Embedding", "Data")
    base_data_dir = "G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/"
    # base_data_dir = os.path.join(".", "sample_data_output") # For local testing

    for example_subdir_part in ["protein.sequences.v12.0.fa", "uniprot_sprot.fasta", "ProteinNegativeAssociations", "uniref100", "dask_checkpoints"]:
        os.makedirs(os.path.join(base_data_dir, example_subdir_part), exist_ok=True)
    # These two were creating directories named after files, let's ensure they are just directories
    os.makedirs(os.path.join(base_data_dir, "protein.sequences.v12.0.fa"), exist_ok=True)
    os.makedirs(os.path.join(base_data_dir, "uniprot_sprot.fasta"), exist_ok=True)

    db_ops = DatabaseOps(DB_CONFIG)
    create_sequences_table_sql = f"""CREATE TABLE IF NOT EXISTS sequences (id INT AUTO_INCREMENT PRIMARY KEY, identifier VARCHAR(255) NOT NULL, source VARCHAR(50), length INT, sequence LONGTEXT, description TEXT, UNIQUE KEY unique_identifier (identifier));"""
    create_uniprot_pdb_table_sql = f"""CREATE TABLE IF NOT EXISTS uniprot_pdb (id INT AUTO_INCREMENT PRIMARY KEY, source VARCHAR(255) NOT NULL, target VARCHAR(50) NOT NULL, INDEX uniprot_idx (source), INDEX pdb_idx (target), UNIQUE KEY unique_mapping (source, target));"""
    create_negative_interactions_table_sql = f"""CREATE TABLE IF NOT EXISTS negative_interactions (id INT AUTO_INCREMENT PRIMARY KEY, source VARCHAR(255), target VARCHAR(255), detection TEXT, first_author VARCHAR(255), publication VARCHAR(255), ncbi_tax_id_source VARCHAR(50), ncbi_tax_id_target VARCHAR(50), interaction_type VARCHAR(255), datasource_name VARCHAR(100), confidence VARCHAR(255));"""

    logger.info("--- Example 1: Parse FASTA and Insert to DB ---")
    if db_ops.execute_ddl(create_sequences_table_sql):
        fasta_dir_ex1 = os.path.join(base_data_dir, "protein.sequences.v12.0.fa")
        string_fasta_file = os.path.join(fasta_dir_ex1, f"dummy_protein_seqs{run_suffix}.fa")
        logger.info(f"Creating/Overwriting dummy FASTA file: {string_fasta_file}")  # Always overwrite
        with open(string_fasta_file, "w") as f:
            f.write(f">seq1{run_suffix} protein one\nACGTACGTACGTNNNN\n")
            f.write(f">seq2{run_suffix} protein two\nTTTTGGGGTTTTCCCC\n")
        fasta_records: List[Tuple[Any, ...]] = []
        batch_size = 100
        for record in tqdm(FileParsingOps.parse_fasta_efficiently(string_fasta_file), desc="Parsing STRING FASTA"):
            fasta_records.append((record['identifier'], "STRING", record['length'], record['sequence'], record['description']))
            if len(fasta_records) >= batch_size: db_ops.insert_data_batch("sequences", ["identifier", "source", "length", "sequence", "description"], fasta_records); fasta_records = []
        if fasta_records: db_ops.insert_data_batch("sequences", ["identifier", "source", "length", "sequence", "description"], fasta_records)
    logger.info("FASTA parsing and DB insertion example finished.")

    logger.info("\n--- Example 2: Parse UniProt FASTA ---")
    fasta_dir_ex2 = os.path.join(base_data_dir, "uniprot_sprot.fasta")
    uniprot_fasta_file = os.path.join(fasta_dir_ex2, f"dummy_uniprot{run_suffix}.fasta")
    logger.info(f"Creating/Overwriting dummy FASTA file: {uniprot_fasta_file}")  # Always overwrite
    with open(uniprot_fasta_file, "w") as f:
        f.write(
            f">sp|P12345{run_suffix}|TEST_HUMAN Test Protein OS=Homo sapiens\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR\n")
    parsed_proteins = FileParsingOps.parse_fasta_to_dict(uniprot_fasta_file, default_source="UNIPROT")
    if parsed_proteins:
        logger.info(f"Parsed {len(parsed_proteins)} sequences from {uniprot_fasta_file}.")
    else:
        logger.info(f"No sequences parsed from {uniprot_fasta_file}.")

    logger.info("\n--- Example 3: Parse TSV (UniProt-PDB) ---")
    if db_ops.execute_ddl(create_uniprot_pdb_table_sql):
        uniprot_pdb_tsv = os.path.join(base_data_dir, f"dummy_uniprot_pdb{run_suffix}.tsv")
        logger.info(f"Creating/Overwriting dummy TSV file: {uniprot_pdb_tsv}")  # Always overwrite
        with open(uniprot_pdb_tsv, "w") as f:
            f.write("h1\th2\n#c\n")  # Header lines
            f.write(f"UniProtID1{run_suffix}\tPDB1{run_suffix};PDB2{run_suffix}\n")
            f.write(f"UniProtID2{run_suffix}\tPDB3{run_suffix}\n")
        interactions: List[Tuple[Any, ...]] = []
        batch_size = 100
        for row in tqdm(FileParsingOps.parse_generic_tsv(uniprot_pdb_tsv, skip_header_lines=2), desc="Parsing UniProt-PDB TSV"):
            if len(row) == 2:
                uid, pids_str = row[0], row[1]
                for pid in pids_str.split(';'):
                    if uid and pid: interactions.append((uid.strip(), pid.strip()))
            if len(interactions) >= batch_size: db_ops.insert_data_batch("uniprot_pdb", ["source", "target"], interactions); interactions = []
        if interactions: db_ops.insert_data_batch("uniprot_pdb", ["source", "target"], interactions)
    logger.info("UniProt-PDB TSV parsing and DB insertion example finished.")

    logger.info("\n--- Example 4: Parse MITAB ---")
    if db_ops.execute_ddl(create_negative_interactions_table_sql):
        mitab_dir = os.path.join(base_data_dir, "ProteinNegativeAssociations")
        mitab_file = os.path.join(mitab_dir, f"dummy_neg_interactions{run_suffix}.mitab")
        logger.info(f"Creating/Overwriting dummy MITAB file: {mitab_file}")  # Always overwrite
        with open(mitab_file, "w") as f:
            f.write("src\ttgt\tdet\taut\tpub\ttaxSrc\ttaxTgt\tintType\tdbSrc\tconf\n")
            f.write(f"uniprot:P12345{run_suffix}\tuniprot:Q67890{run_suffix}\tpsi-mi:\"MI:0007\"\tAuth(24)\tpmid:123\t9606\t9606\tpsi-mi:\"MI:0407\"\tIntAct\tscore:0.8\n")
            f.write(f"uniprot:PABCDE{run_suffix}\tuniprot:QFGHIJ{run_suffix}\tpsi-mi:\"MI:0004\"\tSmith(23)\tpmid:456\t10090\t10090\tpsi-mi:\"MI:0915\"\tBioGRID\tconf:high\n")
        cols = ["source", "target", "detection", "first_author", "publication", "ncbi_tax_id_source", "ncbi_tax_id_target", "interaction_type", "datasource_name", "confidence"]
        mitab_recs: List[Tuple[Any, ...]] = []
        batch_size = 100
        for row_data in tqdm(FileParsingOps.parse_generic_tsv(mitab_file, delimiter='\t', skip_header_lines=1), desc="Parsing MITAB"):
            if len(row_data) >= len(cols): mitab_recs.append(tuple(row_data[i].strip() for i in range(len(cols))))
            if len(mitab_recs) >= batch_size: db_ops.insert_data_batch("negative_interactions", cols, mitab_recs); mitab_recs = []
        if mitab_recs: db_ops.insert_data_batch("negative_interactions", cols, mitab_recs)
    logger.info("MITAB parsing and DB insertion example finished.")

    logger.info("\n--- Example 5: Parse UniRef FASTA to TSV ---")
    uniref_dir = os.path.join(base_data_dir, "uniref100")
    uniref_fasta_input = os.path.join(uniref_dir, f"dummy_uniref{run_suffix}.fasta")
    uniref_tsv_output = os.path.join(uniref_dir, f"dummy_uniref_parsed{run_suffix}.tsv")
    num_dummy_uniref_records = 20000;
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    logger.info(f"Creating/Overwriting dummy UniRef FASTA: {uniref_fasta_input} ({num_dummy_uniref_records} records).")  # Always overwrite
    with open(uniref_fasta_input, "w") as f:
        for i in range(num_dummy_uniref_records):
            seq_len = random.randint(50, 500)
            seq = "".join(random.choices(amino_acids, k=seq_len))
            f.write(f">UniRef100_A0A{i:06X}{run_suffix} Prot{i} n=1 Tax=Test OX=0 RepID=REP{i:06X}{run_suffix}\n{seq}\n")
    uniref_header = ["UniRef_ID", "Organism", "Sequence_Length", "Tax_ID", "Rep_ID", "Sequence"]
    try:
        with open(uniref_tsv_output, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=uniref_header, delimiter='\t')
            writer.writeheader();
            count = 0
            for rec in tqdm(FileParsingOps.parse_uniref_fasta_to_tsv_rows(uniref_fasta_input), desc="Parsing UniRef to TSV"):
                writer.writerow(rec);
                count += 1
            logger.info(f"UniRef FASTA to TSV complete. {count} records to {uniref_tsv_output}")
    except IOError as e:
        logger.error(f"Error writing UniRef TSV to '{uniref_tsv_output}': {e}")

    logger.info("\n--- Example 6: Dask Workflow ---")
    large_tsv_filepath = uniref_tsv_output
    if not os.path.exists(large_tsv_filepath) or os.path.getsize(large_tsv_filepath) < 1000:
        logger.error(f"Dask input TSV {large_tsv_filepath} missing/small. Ensure Ex5 ran.");
        return

    seq_col = "Sequence";
    dask_blocksize = '1MB'
    checkpoint_base_dir = os.path.join(base_data_dir, "dask_checkpoints")
    pq_dir1 = os.path.join(checkpoint_base_dir, f"sequences_parquet{run_suffix}")
    pq_dir2 = os.path.join(checkpoint_base_dir, f"processed_sequences_parquet{run_suffix}")

    ddf_seqs = ProteinOps.load_sequences_with_dask(large_tsv_filepath, seq_col, blocksize=dask_blocksize)

    if ddf_seqs is not None and ddf_seqs.npartitions > 0:
        logger.info(f"Dask DF from TSV head:\n{ddf_seqs.head()}")
        if DaskOps.save_dask_dataframe_to_parquet(ddf_seqs, pq_dir1, overwrite=True, compression='gzip'):
            logger.info(f"Initial Dask DF saved to Parquet: {pq_dir1}")
        else:
            logger.error(f"Failed to save initial Dask DF to {pq_dir1}. Workflow cannot continue."); return
    else:
        logger.error("Failed to load data from TSV for Dask or Dask DF empty. Exiting."); return

    loaded_ddf = DaskOps.load_dask_dataframe_from_parquet(pq_dir1, columns=[seq_col])
    if loaded_ddf is None or loaded_ddf.npartitions == 0:
        logger.error("Failed to load Dask DF from Parquet or it's empty. Exiting.");
        return

    meta_proc: Dict[str, Any] = {'Sequence': loaded_ddf['Sequence'].dtype, 'Sequence_Length': 'int64', 'Sequence_Edges': 'object'}
    logger.info(f"Applying partition processing. Input partitions: {loaded_ddf.npartitions}, Meta: {meta_proc}")
    result_ddf = loaded_ddf.map_partitions(BioWorkflowOps.dask_partition_processor, meta=meta_proc)

    logger.info("Computing Dask result...")
    try:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            computed_pdf = result_ddf.compute()

        if computed_pdf.empty:
            logger.warning("Computed Dask result (computed_pdf) is empty. Skipping save to final Parquet.")
        else:
            logger.info(f"Computed Dask result head:\n{computed_pdf.head()}")
            num_out_parts = max(1, min(loaded_ddf.npartitions, (len(computed_pdf) // 10000) + 1))
            final_ddf = dd.from_pandas(computed_pdf, npartitions=num_out_parts)

            # Define the explicit Arrow schema for the processed data
            # This must match the actual structure of final_ddf columns and their order
            # Ensure the 'Sequence' dtype matches what Dask infers or what pandas produces
            # For 'object' dtype string columns, pa.string() is usually correct.
            # If loaded_ddf['Sequence'].dtype was specific like pd.StringDtype(), its .to_arrow() would be better.
            # For simplicity, assuming string for 'Sequence' column in Parquet.

            sequence_arrow_type = pa.string()  # Default to string
            # Attempt to get a more precise arrow type from the Dask series if possible
            # This is a bit more robust if the Dask series has a specific pandas extension dtype
            try:
                if hasattr(final_ddf['Sequence'].dtype, 'to_arrow'):
                    sequence_arrow_type = final_ddf['Sequence'].dtype.to_arrow()
                elif final_ddf['Sequence'].dtype == object:  # common for strings
                    sequence_arrow_type = pa.string()

            except AttributeError:  # Fallback if dtype doesn't have to_arrow
                sequence_arrow_type = pa.string()

            final_arrow_schema = pa.schema([
                pa.field('Sequence', sequence_arrow_type),
                pa.field('Sequence_Length', pa.int64()),
                pa.field('Sequence_Edges', pa.list_(
                    pa.struct([
                        pa.field('from_node', pa.string()),  # field() is good practice
                        pa.field('to_node', pa.string()),
                        pa.field('weight', pa.float64())
                    ])
                ))
            ])
            # Verify column order:
            # final_ddf_columns = final_ddf.columns.tolist()
            # schema_columns = [f.name for f in final_arrow_schema]
            # if final_ddf_columns != schema_columns:
            #    logger.error(f"Column order mismatch! DF: {final_ddf_columns}, Schema: {schema_columns}")
            #    # Potentially reorder schema or df columns here if necessary, though map_partitions meta should ensure order.

            logger.info(f"Saving processed Dask DF ({len(computed_pdf)} rows, {final_ddf.npartitions} parts) to Parquet: {pq_dir2} with explicit schema.")
            if DaskOps.save_dask_dataframe_to_parquet(final_ddf, pq_dir2, overwrite=True, compression='gzip', schema=final_arrow_schema):  # Pass schema
                logger.info(f"Loading 'Sequence_Edges' from final Parquet: {pq_dir2}")
                seq_edges_ddf = DaskOps.load_dask_dataframe_from_parquet(pq_dir2, columns=["Sequence_Edges"])
                if seq_edges_ddf is not None:
                    logger.info(f"Sequence_Edges Dask DF head:\n{seq_edges_ddf.head()}")
                    logger.info(f"Partitions in Sequence_Edges DDF: {seq_edges_ddf.npartitions}")
            else:
                logger.error(f"Failed to save final processed Dask DF to {pq_dir2}.")
    except Exception as e:
        logger.error(f"Error during Dask computation/final saving: {e}", exc_info=True)

    logger.info("Dask workflow example finished.")
    logger.info("Main example workflow finished.")

def load_uniprot_database():
    logger.info("Starting workflow...")
    base_data_dir = "G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/"
    db_ops = DatabaseOps(DB_CONFIG)
    create_sequences_table_sql = f"""CREATE TABLE IF NOT EXISTS uniprot_sequences (id INT AUTO_INCREMENT PRIMARY KEY, identifier VARCHAR(255) NOT NULL, source VARCHAR(50), length INT, sequence LONGTEXT, description TEXT, UNIQUE KEY unique_identifier (identifier));"""
    logger.info("--- Parse FASTA and Insert to DB ---")
    if db_ops.execute_ddl(create_sequences_table_sql):
        fasta_dir_ex1 = os.path.join(base_data_dir, "uniprot_sprot.fasta")
        logger.info(f"Creating/Overwriting FASTA file: {fasta_dir_ex1}")
        fasta_records: List[Tuple[Any, ...]] = []
        batch_size = 100
        for record in tqdm(FileParsingOps.parse_fasta_efficiently(fasta_dir_ex1), desc="Parsing UNIPROT FASTA"):
            fasta_records.append((record['identifier'], "UNIPROT", record['length'], record['sequence'], record['description']))
            if len(fasta_records) >= batch_size: db_ops.insert_data_batch("uniprot_sequences", ["identifier", "source", "length", "sequence", "description"], fasta_records); fasta_records = []
        if fasta_records: db_ops.insert_data_batch("uniprot_sequences", ["identifier", "source", "length", "sequence", "description"], fasta_records)
    logger.info("FASTA parsing and DB insertion finished.")


if __name__ == "__main__":
    from dask.diagnostics import ProgressBar

    # tests()
    # load_uniprot_database()