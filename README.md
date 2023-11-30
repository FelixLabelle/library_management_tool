# Digital Library Management Tool

## Overview

This project is designed to make text documents searchable using two  scripts: `library_parser` and `library_search`. 

These scripts enable search functionality through a command-line interface (CLI). 

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.10
- Required Python libraries (specified in `env.yml`)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/FelixLabelle/library_management_tool.git
    cd library_management_tool
    ```

2. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### `library_parser`

The `library_parser` script processes text documents. Use the following command-line arguments:

```bash
python library_parser.py -r [path/to/PDFs] -t [versioning_tag] --write_count [write_count_value] --passage_size [passage_size_value] --tokenizer [tokenizer_name]
```

The arguments are:

    -r or --root_dir: Path to the directory where PDFs are stored.
    -t or --tag: Versioning tag used to track different indices.
    --write_count: Save temporary files frequency (default: 20, can be adjusted for speed).
    --passage_size: Size of passages for retrieval (default: 250).
    --tokenizer: Sentence tokenizer to be used (default: "en_core_web_sm").
### library_search

The library_search script uses the output of library_parser for CLI-based searches. Use the following command-line arguments:

```bash
python library_search.py -f [dataset_filename] --search_type [search_type] --model_name [model_name] --top_k [top_k_value]
```
The arguments are:

    -f or --dataset_filename: Path to the dataset file.
    --search_type: Type of search to perform (ann or exact, default: exact).
    --model_name: Name of the model (sentence transformer models only supported currently, default: 'all-mpnet-base-v2').
    --top_k: Output k best results (default: 10).
	
## Additional information

Currently project only supports PDFs, looking to expand to other formats like EPUB eventually.