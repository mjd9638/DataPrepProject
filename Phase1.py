"""
@Authors: Ty Minnick. Mike Dortz, Maham Imtiaz, Deryke Tang
Description: This file contains preliminary data mining for phase 1 of the project.
Date: 9/9/2021
"""

import sys
import pandas as pd


def merge_databases(names, akas, basics, ratings):
    """
    Merges the databases together.
    """
    final_df = pd.merge(basics, ratings, on='tconst', how='left')
    akas.rename(columns={'titleId': 'tconst'}, inplace=True)
    print(akas.columns, '\n')
    final_df = pd.merge(final_df, akas, on='tconst', how='left')
    print(final_df.columns, '\n')
    names = names.explode('knownForTitles')
    names.rename(columns={'knownForTitles': 'tconst'}, inplace=True)
    final_df = pd.merge(final_df, names, on='tconst', how='left')
    print(final_df.head())


def main(argv):
    #pd.set_option('display.max_columns', None)
    name_basics_df = pd.read_csv("name.basics.tsv", sep='\t', dtype=str, converters={'knownForTitles': lambda x: x.split(',')})
    title_akas_df = pd.read_csv("title.akas.tsv", sep='\t', dtype=str)
    title_basics_df = pd.read_csv("title.basics.tsv", sep='\t', dtype=str)
    title_ratings_df = pd.read_csv("title.ratings.tsv", sep='\t')
    merge_databases(name_basics_df, title_akas_df, title_basics_df, title_ratings_df)


if __name__ == '__main__':
    main(sys.argv[1:])


