<!-- # readbiomed-lbd
The repository for the paper <>

## Repo structure
- 01_data
  - tbd 
- 02_code 
  - 0201_process_corpus
  - 0202_create_annotator
  - 0203_process_annotation
  - 0204_generate_graph
  - 0205_graph_embedding_models
  - 0206_analyze_graph_statistics
  - 0207_analyze_predictions -->

# Graph embedding-based link prediction for literature-based discovery in Alzheimer’s Disease

This repository is for the paper under review in Journal of Biomedical Informatics.

> Authors: Yiyuan Pu, Daniel Beck, Karin Verspoor. 

## Prerequisites

- Python 3.8 

## Directory Structure

- `./00_data` directory contains input files
- `./01_collect_corpus` directory contains scripts for collecting corpus
- `./02_create_annotator` directory contains scripts for creating AD-specific annotators
- `./03_process_annotation` directory contains scripts for processing annotations
- `./04_generate_graph` directory contains scripts for generating the AD knowledge graph
- `./05_infer_knowledge` directory contains scripts for predicting putative links with graph embedding models
- `./06_analyze_graph` directory contains scripts for analyzing graph statistics
- `./07_analyze_predictions` directory contains scripts for analyzing outputs from link prediction models

<!-- ## Usage
1. Download the repo from [The Pathogen Annotator](https://github.com/READ-BioMed/readbiomed-pathogen-annotator)
2. Download the owl files from the `./00_data` directory
3. Prepare `sub_rel_obj_pyear_edat_pmid_sent_id_sent.tsv.gz` file and place it into the `./data/SemMedDB` directory
4. Download SemRepped [CORD-19](https://ii.nlm.nih.gov/SemRep_SemMedDB_SKR/COVID-19/index.shtml) dataset and extract files into `./data/cord-19 directory`
5. Prepare SemMedDB and CORD-19 data using the `./preprocessing/run.sh` file
6. Run Python notebooks in the `./filtering` directory
7. Run Python notebooks in the `./models` directory -->

## Contact
Karin Verspoor (`karin.verspoor (at) rmit.edu.au`) or Yiyuan Pu (`yiyuanp1 (at) student.unimelb.edu.au`)

