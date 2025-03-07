# DATA407_HousePriceAnalysis
This is a project from DATA407.  
This project is an analysis of house price in Japan.

## Setup
For first-time setup, follow the instructions in the [Setup Guide](docs/setup_guide.md).

## Restarting Development
If you have already set up your environment, follow these steps to restart development:

```bash
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Stopping Development
To stop Jupyter Lab, press `Ctrl + C` twice in the terminal.  
Then deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

The following is the directory structure of the project:

```
DATA407_HousePriceAnalysis/
├── data/
│   ├── encoded/
│   ├── processed/
│   ├── raw/
├── docs/
│   ├── setup_guide.md
├── notebooks/
│   ├── 00_process_encoding.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
├── reports/
│   ├── final_report.md
├── results/
│   ├── figures/
│   ├── tables/
├── venv/
├── .gitignore
├── README.md
├── requirements.txt
```

### **Directory Descriptions**
- **`data/`**: Contains raw, processed, and encoded datasets.
  - **`raw/`**: Original unprocessed data files.
  - **`processed/`**: Cleaned and transformed data ready for analysis.
  - **`encoded/`**: Data converted to UTF-8 encoding.

- **`docs/`**: Documentation related to the project.
  - `setup_guide.md`: Instructions for setting up the project.

- **`notebooks/`**: Jupyter notebooks used for data processing and analysis.
  - `00_process_encoding.ipynb`: Encoding and preprocessing of data.
  - `01_data_cleaning.ipynb`: Data cleaning steps.
  - `02_eda.ipynb`: Exploratory data analysis.

- **`reports/`**: Contains reports generated from the analysis.
  - `final_report.md`: The main report summarizing findings.

- **`results/`**: Stores figures and tables generated from analysis.
  - `figures/`: Visualizations and plots.
  - `tables/`: Summary tables and other structured results.

- **`venv/`**: Virtual environment for dependency management.

- **`.gitignore`**: Specifies files and directories to be ignored by Git.

- **`README.md`**: Main project documentation (this file).

- **`requirements.txt`**: List of dependencies required for the project.

