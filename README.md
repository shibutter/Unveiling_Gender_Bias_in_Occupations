# Unveiling Gender Bias in Occupations: A Comparative Analysis of GPT-3.5 and Llama 2 in the Generation of Dutch Short Stories

## Repository Structure
- `generate_data.ipynb`: notebook for generating datasets by creating prompts and stories using GPT-3.5.
- `data_analysis.ipynb`: notebook for analysing occupation datasets, comparing unique occupations, and generating statistics.
- `data_preprocess_analysis.ipynb`: notebook for preprocessing the datasets, including text cleaning and standardizing gender and occupation fields.
- `fighting_words.py`: this code is adapted from the implementation of the Fightin' Words paper by Monroe, Colaresi, & Quinn (2008), available at [FightingWords GitHub repository](https://github.com/jmhessel/FightingWords). Minor adjustments are made to extract the adjectives and verbs.
- `generate_fw_data.ipynb`: notebook for processing datasets to extract adjectives and verbs, divided by genres using the `fighting_words.py' script.
- `fw_analysis.ipynb`: notebook for comparing "Fightin' Word Scores" between GPT-3.5 and Llama 2 using scatter plots.
- `human_data_analysis.ipynb`: notebook for analysing human survey data and comparing it with GPT-3.5 and Llama 2 datasets.
- `data`: this directory includes both raw and processed data for GPT-3.5 and Llama 2, along with a `fw_data` directory specifically for the Fightin' Words data.
- `plots`: this directory contains the plots that are also included in the Appendix.

