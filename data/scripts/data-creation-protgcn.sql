-- =============================================================================
-- Script Setup
-- =============================================================================
USE pdi;

-- =============================================================================
-- DDL: Table and View Definitions
-- =============================================================================

-- Table for negative interactions
-- Consolidating the two similar definitions for 'negative' and 'negative_interactions'
-- Using TEXT for many columns; consider VARCHAR(N) for better performance on IDs/codes if applicable.
CREATE TABLE IF NOT EXISTS pdi.negative_interactions (
    edge_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    source TEXT NULL, -- Consider VARCHAR(255) or appropriate length if it's an ID
    target TEXT NULL, -- Consider VARCHAR(255) or appropriate length if it's an ID
    alt_source TEXT NULL,
    alt_target TEXT NULL,
    alias_source TEXT NULL,
    alias_target TEXT NULL,
    detection TEXT NULL,
    first_author TEXT NULL,
    publication TEXT NULL,
    ncbi_tax_id_source VARCHAR(50) NULL, -- Example of using VARCHAR for IDs
    ncbi_tax_id_target VARCHAR(50) NULL, -- Example of using VARCHAR for IDs
    interaction_type TEXT NULL,
    datasource_name TEXT NULL,
    datasource_id TEXT NULL,
    confidence TEXT NULL,
    expansion TEXT NULL,
    bio_role_source TEXT NULL,
    bio_role_target TEXT NULL,
    exp_role_source TEXT NULL,
    exp_role_target TEXT NULL,
    type_source TEXT NULL,
    type_target TEXT NULL,
    gene_reference_source TEXT NULL,
    gene_reference_target TEXT NULL,
    gene_reference_interaction TEXT NULL,
    annotation_source TEXT NULL,
    annotation_target TEXT NULL,
    annotation_interaction TEXT NULL,
    ncbi_tax_id_host VARCHAR(50) NULL, -- Example of using VARCHAR for IDs
    creation_date VARCHAR(20) NULL, -- Consider DATE or DATETIME if appropriate
    update_date VARCHAR(20) NULL,   -- Consider DATE or DATETIME if appropriate
    checksum_source TEXT NULL,
    checksum_target TEXT NULL,
    checksum_interaction TEXT NULL,
    type_interaction TEXT NULL
    -- Consider adding indexes on source, target, ncbi_tax_id_source, ncbi_tax_id_target, datasource_id etc.
    -- EXAMPLE: INDEX idx_negint_source (source(255)), -- Prefix index if source is TEXT
    -- EXAMPLE: INDEX idx_negint_target (target(255))   -- Prefix index if target is TEXT
);

-- View for BioGRID gene IDs (created once)
-- This pre-processes the SUBSTRING logic, which is good for performance in subsequent queries.
CREATE OR REPLACE VIEW pdi.biogrid_gene_ids AS
SELECT
    SUBSTRING(mitab.`Interactor A`, 23, 1000) AS GeneID_Interactor_A, -- Assuming the first 22 chars are a prefix like 'entrez gene/locuslink:'
    SUBSTRING(mitab.`Interactor B`, 23, 1000) AS GeneID_Interactor_B
FROM
    pdi.biogrid_to_uniprot_positive_interactions mitab;

select count(*) from russell_lab_to_uniprot_negative_interaction;

select UNIPROTID_A, UNIPROTID_B from pdi.biogrid_to_uniprot_positive_interactions;

select UNIPROTID_A, UNIPROTID_B from pdi.russell_lab_to_uniprot_negative_interaction;

select * from uniprot_to_entrez;

-- Table for mapping BioGRID Gene IDs to UniProt IDs
-- Using the more efficient LEFT JOIN version.
-- This table is rebuilt each time the script runs this section.
DROP TABLE IF EXISTS pdi.biogrid_to_uniprot;
CREATE TABLE pdi.biogrid_to_uniprot AS
SELECT
    b.GeneID_Interactor_A,
    b.GeneID_Interactor_B,
    u1.UNIPROT_ID AS UNIPROTID_A,
    u2.UNIPROT_ID AS UNIPROTID_B
FROM
    pdi.biogrid_gene_ids b -- Use the view
LEFT JOIN
    pdi.uniprot_to_entrez u1 ON b.GeneID_Interactor_A = u1.GENEID
LEFT JOIN
    pdi.uniprot_to_entrez u2 ON b.GeneID_Interactor_B = u2.GENEID;
-- Consider adding indexes:
-- ALTER TABLE pdi.biogrid_to_uniprot ADD INDEX idx_btu_geneA (GeneID_Interactor_A);
-- ALTER TABLE pdi.biogrid_to_uniprot ADD INDEX idx_btu_geneB (GeneID_Interactor_B);
-- ALTER TABLE pdi.biogrid_to_uniprot ADD INDEX idx_btu_uniprotA (UNIPROTID_A);
-- ALTER TABLE pdi.biogrid_to_uniprot ADD INDEX idx_btu_uniprotB (UNIPROTID_B);


-- Modifying protein_sequences table (assuming it exists)
-- Ensure the identifier format is consistent for SUBSTRING to work as expected.
-- The length 1000 for chain_id seems large if chain_id is typically 1-2 characters. VARCHAR(10) is kept.
-- This ALTER TABLE statement will fail if the column already exists.
-- To make it idempotent, you might need to drop it first if it could exist with a different definition.
-- However, GENERATED COLUMNS cannot be easily dropped and re-added if their definition changes without more complex DDL.
-- For now, assume it's run once or the table is in a state where this can be added.
-- A safer approach if re-runnable: check if column exists, or use a procedure.
-- For this script, it will attempt to add it. If it fails due to existing column, the script might halt depending on SQL client.
ALTER TABLE pdi.protein_sequences
ADD COLUMN IF NOT EXISTS chain_id VARCHAR(10) GENERATED ALWAYS AS (SUBSTRING(identifier, 6, 10)) STORED;
-- NOTE: Changed SUBSTRING length to 10 to match VARCHAR(10). Adjust if chain_id can be longer up to original 1000 or if identifier format implies different substring indices.


-- =============================================================================
-- DML: Data Modification (Use with caution)
-- =============================================================================

-- Deleting sequences not sourced from PDB. This is a destructive operation.
-- Commented out by default for safety. Uncomment to run.
/*
SELECT COUNT(*) AS count_before_delete FROM pdi.protein_sequences WHERE source NOT LIKE "%PDB%";
DELETE FROM pdi.protein_sequences WHERE source NOT LIKE "%PDB%";
SELECT COUNT(*) AS count_after_delete FROM pdi.protein_sequences WHERE source NOT LIKE "%PDB%";
*/

-- =============================================================================
-- Cleanup `DROP` Statements (Optional - use with caution)
-- =============================================================================

DROP VIEW IF EXISTS pdi.gene_mapping; -- This view was mentioned in a drop but not created in the provided script.
-- DROP TABLE IF EXISTS pdi.pdb_chains; -- This is destructive if pdb_chains is a primary data table. Commenting out.


-- =============================================================================
-- SELECT Queries: Data Exploration and Analysis
-- =============================================================================

-- General Counts and Basic Exploration
SELECT 'Count of negative_interactions' AS query_description, COUNT(*) AS total_count FROM pdi.negative_interactions;
SELECT 'Sample from negative_interactions (limit 1000)' AS query_description, edge_id, source, target, datasource_name FROM pdi.negative_interactions LIMIT 1000; -- Selected specific columns

SELECT 'Count of sequences' AS query_description, COUNT(*) AS total_count FROM pdi.sequences;
SELECT 'Count of sequences from UNIPROT source' AS query_description, COUNT(*) AS uniprot_source_count FROM pdi.sequences WHERE source LIKE '%UNIPROT%'; -- LIKE '%...%' is slow
SELECT 'Sample PDB sequence for 1A2B' AS query_description, identifier, source, length, description FROM pdi.sequences WHERE sequences.source = 'PDB' AND identifier LIKE '%1A2B%' LIMIT 10; -- Selected columns

SELECT 'Count from pp_pathways_ppi' AS query_description, COUNT(*) AS total_count FROM pdi.pp_pathways_ppi; -- Assuming this table exists

SELECT 'Sample from protein_sequences (limit 1000)' AS query_description, sequence_id, identifier, source, length FROM pdi.protein_sequences LIMIT 1000; -- Selected columns

SELECT 'UniProt-PDB mappings for Q6GZX4' AS query_description, source, target FROM pdi.uniprot_pdb WHERE source LIKE '%Q6GZX4%' LIMIT 100; -- LIKE '%...%' is slow

SELECT 'Sample from biogrid_all_4_4_243_mitab (limit 1000)' AS query_description, `Interactor A`, `Interactor B`, `Interaction Types` FROM pdi.biogrid_all_4_4_243_mitab LIMIT 1000; -- Selected columns

SELECT 'Distinct UniProt IDs from pdb_chains' AS query_description, COUNT(DISTINCT SP_PRIMARY) as distinct_uniprot_ids FROM pdi.pdb_chains;
SELECT 'Sample UniProt IDs from pdb_chains' AS query_description, DISTINCT SP_PRIMARY AS UNIPROT_ID FROM pdi.pdb_chains LIMIT 100;


-- BioGRID to UniProt Mapping Exploration (using the biogrid_gene_ids view and biogrid_to_uniprot table)
SELECT 'Sample from uniprot_to_entrez (limit 100)' AS query_description, UNIPROT_ID, GENEID FROM pdi.uniprot_to_entrez LIMIT 100;

-- Joining BioGRID Gene IDs with UniProt-Entrez mapping
SELECT
    'BioGRID Interactor A mapped to UniProt (Sample)' AS query_description,
    b.GeneID_Interactor_A,
    b.GeneID_Interactor_B,
    e.UNIPROT_ID
FROM
    pdi.biogrid_gene_ids b
INNER JOIN
    pdi.uniprot_to_entrez e ON b.GeneID_Interactor_A = e.GENEID
LIMIT 1000;

SELECT
    'BioGRID Interactor B mapped to UniProt (Sample)' AS query_description,
    b.GeneID_Interactor_A,
    b.GeneID_Interactor_B,
    e.UNIPROT_ID
FROM
    pdi.biogrid_gene_ids b
INNER JOIN
    pdi.uniprot_to_entrez e ON b.GeneID_Interactor_B = e.GENEID
LIMIT 1000;

-- Exploring the created biogrid_to_uniprot table
SELECT 'Count from biogrid_to_uniprot' AS query_description, COUNT(*) AS total_mappings FROM pdi.biogrid_to_uniprot;
SELECT 'Sample from biogrid_to_uniprot with successful A mapping' AS query_description, * FROM pdi.biogrid_to_uniprot WHERE UNIPROTID_A IS NOT NULL LIMIT 100;
SELECT 'Sample from biogrid_to_uniprot with successful B mapping' AS query_description, * FROM pdi.biogrid_to_uniprot WHERE UNIPROTID_B IS NOT NULL LIMIT 100;

-- This complex join was provided, likely for finding interactions where both interactors can be mapped through a common gene ID system.
-- It seems to try to bridge interactions. It's potentially very expensive.
-- The original query:
-- select * from (select * from pdi.biogrid JOIN (SELECT DISTINCT UNIPROT_ID, GENEID FROM pdi.uniprot_to_entrez) as t3 ON t3.GENEID = pdi.biogrid.GeneID_Interactor_A) as t4 JOIN pdi.biogrid ON t4.GENEID = pdi.biogrid.GeneID_Interactor_B;
-- This can be rewritten using the biogrid_to_uniprot table if the goal is to find UniProt pairs.
-- If the goal is to find A->Gene1, B->Gene2, and then find other interactions where Gene1 is an interactor and Gene2 is an interactor:
-- The query seems to imply: find (A,B) from biogrid. Map A to GeneID (t3.GENEID). Then find other (C,D) from biogrid where C is t3.GENEID.
-- This doesn't quite make sense with the final join `ON t4.GENEID = pdi.biogrid.GeneID_Interactor_B`.
-- Assuming `t4.GENEID` refers to `GeneID_Interactor_A`'s mapping.
-- Let's re-interpret or simplify if possible. If the goal is to find A-B pairs where both A and B map to UniProt IDs:
SELECT
    'BioGRID interactions where both interactors map to UniProt (Sample)' AS query_description,
    btu.GeneID_Interactor_A,
    btu.UNIPROTID_A,
    btu.GeneID_Interactor_B,
    btu.UNIPROTID_B
FROM
    pdi.biogrid_to_uniprot btu
WHERE
    btu.UNIPROTID_A IS NOT NULL AND btu.UNIPROTID_B IS NOT NULL
LIMIT 1000;


-- Russell Lab Negative Interactions
SELECT 'Count from russell_lab_to_uniprot_negative_interaction' AS query_description, COUNT(*) FROM pdi.russell_lab_to_uniprot_negative_interaction; -- Assuming this table exists

-- Extracting UniProt IDs from Russell Lab negative interactions for 'true' interactions
SELECT
    'Russell Lab True Negative Interactions (UniProt IDs)' AS query_description,
    SUBSTRING(rlni.`source`, 11, 1000) AS UNIPROTID_A, -- Assuming 'uniprotkb:' prefix (10 chars)
    SUBSTRING(rlni.`target`, 11, 1000) AS UNIPROTID_B  -- Assuming 'uniprotkb:' prefix (10 chars)
FROM
    pdi.russell_lab_negative_interactions rlni
WHERE
    rlni.type_interaction LIKE '%true%' -- This implies type_interaction might contain more than just 'true'
LIMIT 1000;


-- PDB Chains Exploration
SELECT 'Sample from pdb_chains for PDB ID 173d' AS query_description, * FROM pdi.pdb_chains WHERE PDB LIKE '%173d%' LIMIT 10; -- LIKE '%...%' is slow
SELECT 'Full sample from pdb_chains (limit 100)' AS query_description, * FROM pdi.pdb_chains LIMIT 100;


-- Protein Sequences Exploration (assuming pdb_id and chain_id columns exist or are generated)
SELECT
    'Protein sequences not from UNIPROT, for identifier 11gs (Sample)' AS query_description,
    sequence_id, identifier, pdb_id, chain_id, source, length, description -- Specify columns
FROM
    pdi.protein_sequences
WHERE
    source NOT LIKE '%UNIPROT%' AND identifier LIKE '%11gs%' -- Both LIKE '%...%' are slow
LIMIT 10;

-- Extracting PDB ID and Chain ID from protein_sequences not sourced from UNIPROT
-- This query assumes 'pdb_id' column might not exist and tries to derive it.
-- If protein_sequences.identifier for PDB entries is like '1A2B_C' (PDBID_CHAIN)
-- The ALTER TABLE already adds a generated chain_id. Let's assume a pdb_id column is also generated or exists.
-- If not, the SUBSTRING logic is:
-- SUBSTRING(ps.identifier, 1, 4) AS pdb_id, -- if identifier starts with PDB ID
-- SUBSTRING(ps.identifier, 6, 1) AS chain_id -- if identifier is PDBID_CHAIN (char at 6th pos)
-- The query below uses the generated chain_id and an assumed pdb_id column
SELECT
    'Non-UNIPROT protein sequences with PDB and Chain IDs (Sample)' AS query_description,
    ps.sequence_id,
    ps.identifier, -- Original identifier
    ps.pdb_id,     -- Assumed or generated PDB ID column
    ps.chain_id,   -- Generated chain_id column
    ps.length,
    ps.sequence
FROM
    pdi.protein_sequences ps
WHERE
    ps.source NOT LIKE '%UNIPROT%' -- Slow if not indexed appropriately or many non-Uniprot entries
LIMIT 1000;

SELECT 'Protein sequences not sourced from PDB (Sample)' AS query_description, sequence_id, identifier, source FROM pdi.protein_sequences WHERE source NOT LIKE '%PDB%' LIMIT 100;

-- PDB to UniProt Mapping Exploration
SELECT 'Sample from pdb_to_uniprot (limit 100)' AS query_description, PDB_ID, CHAIN_ID, UNIPROT_ID FROM pdi.pdb_to_uniprot LIMIT 100;
SELECT 'PDB ID counts in pdb_to_uniprot' AS query_description, COUNT(DISTINCT PDB_ID) as distinct_pdb_ids FROM pdi.pdb_to_uniprot;
SELECT 'UniProt ID counts in pdb_to_uniprot' AS query_description, COUNT(DISTINCT UNIPROT_ID) as distinct_uniprot_ids FROM pdi.pdb_to_uniprot;

-- Frequency of UniProt IDs in pdb_to_uniprot
SELECT
    'Top 20 UniProt IDs by PDB mapping frequency' AS query_description,
    UNIPROT_ID,
    COUNT(UNIPROT_ID) AS frequency
FROM
    pdi.pdb_to_uniprot
GROUP BY
    UNIPROT_ID
ORDER BY
    frequency DESC
LIMIT 20;

-- Joining protein_sequences with pdb_to_uniprot
-- Using explicit INNER JOIN and selecting specific columns
SELECT
    'Protein sequences joined with PDB-UniProt mapping (Sample)' AS query_description,
    ps.sequence_id,
    ps.pdb_id,      -- from protein_sequences
    ps.chain_id,    -- from protein_sequences
    pu.UNIPROT_ID,  -- from pdb_to_uniprot
    ps.length,
    ps.sequence,
    ps.description
FROM
    pdi.protein_sequences ps
INNER JOIN
    pdi.pdb_to_uniprot pu ON ps.pdb_id = pu.PDB_ID AND (ps.chain_id = pu.CHAIN_ID OR pu.CHAIN_ID IS NULL) -- Join on chain if available
LIMIT 1000;
-- The DISTINCT * version is kept below if all columns are truly needed and distinct rows from the join result.
SELECT 'Distinct rows from protein_sequences joined with PDB-UniProt (Sample)' AS query_description, DISTINCT ps.*, pu.*
FROM pdi.protein_sequences ps
INNER JOIN pdi.pdb_to_uniprot pu ON ps.pdb_id = pu.PDB_ID AND (ps.chain_id = pu.CHAIN_ID OR pu.CHAIN_ID IS NULL) -- Consider if chain match is always required
LIMIT 1000; -- LIMIT after DISTINCT can be slow.

SELECT 'Protein sequences not having "PDB" as source (Sample)' AS query_description, sequence_id, identifier, source FROM protein_sequences WHERE source != 'PDB' LIMIT 100; -- More direct than NOT LIKE '%PDB%' if 'PDB' is exact.

SELECT 'Count from biogrid_to_uniprot_positive_interactions' AS query_description, COUNT(*) FROM pdi.biogrid_to_uniprot_positive_interactions; -- Assuming this table exists
SELECT 'Sample from biogrid_to_uniprot_positive_interactions for A0A2A5ATH9' AS query_description, * FROM pdi.biogrid_to_uniprot_positive_interactions WHERE UNIPROTID_A = "A0A2A5ATH9" OR UNIPROTID_B = "A0A2A5ATH9" LIMIT 10;

-- Grouping interactions (GROUP_CONCAT can result in very long strings and has a default limit)
SELECT
    'Grouped interactions by UNIPROTID_A (Sample)' AS query_description,
    UNIPROTID_A,
    GROUP_CONCAT(UNIPROTID_B SEPARATOR ', ') AS Mapped_Bs
FROM
    pdi.biogrid_to_uniprot_positive_interactions -- Assuming this table contains UNIPROTID_A and UNIPROTID_B
GROUP BY
    UNIPROTID_A
LIMIT 100;

SELECT
    'Grouped interactions by UNIPROTID_B (Sample)' AS query_description,
    UNIPROTID_B,
    GROUP_CONCAT(UNIPROTID_A SEPARATOR ', ') AS Mapped_As
FROM
    pdi.biogrid_to_uniprot_positive_interactions
GROUP BY
    UNIPROTID_B
LIMIT 100;

SELECT 'Count of distinct UNIPROTID_A with grouped interactions' AS query_description, COUNT(*) FROM (
    SELECT UNIPROTID_A FROM pdi.biogrid_to_uniprot_positive_interactions GROUP BY UNIPROTID_A
) AS distinct_A_groups;

SELECT 'Count of distinct UNIPROTID_B with grouped interactions' AS query_description, COUNT(*) FROM (
    SELECT UNIPROTID_B FROM pdi.biogrid_to_uniprot_positive_interactions GROUP BY UNIPROTID_B
) AS distinct_B_groups;


-- Specific PDB ID query
SELECT
    'Protein sequence and UniProt mapping for PDB ID 102l' AS query_description,
    ps.sequence_id,
    ps.pdb_id,
    pu.UNIPROT_ID,
    ps.chain_id,
    ps.length,
    ps.sequence,
    ps.description
FROM
    pdi.protein_sequences ps
LEFT JOIN -- Using LEFT JOIN in case a sequence entry doesn't have a UniProt mapping yet
    pdi.pdb_to_uniprot pu ON ps.pdb_id = pu.PDB_ID AND ps.chain_id = pu.CHAIN_ID -- Assuming chain match is important
WHERE
    ps.pdb_id = "102l";

-- Final check queries
SELECT 'Final sample from protein_sequences' AS query_description, sequence_id, pdb_id, chain_id, source, length FROM pdi.protein_sequences LIMIT 10;
select count(*) from pdi.uniprot_sequences;


-- =============================================================================
-- End of Script
-- =============================================================================