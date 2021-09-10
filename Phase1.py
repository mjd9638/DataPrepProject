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
    print(final_df.columns, '\n')
    print(final_df.head())
    final_df.to_csv('final.tsv', sep='\t')


def split_attributes(dataset):
    attributes = ['types', 'attributes', 'genres', 'primaryProfession']
    for each in attributes:
        dataset[each] = dataset[each].str.split(',')
        dataset[each] = dataset.explode(each)
    print(dataset.head())
    dataset.to_csv('final.tsv', sep='\t')


def main(argv):
    #pd.set_option('display.max_columns', None)
    # read in names data set
    print('Reading in names...')
    #name_basics_df = pd.read_csv("name.basics.tsv", sep='\t', dtype=str, converters={'knownForTitles': lambda x: x.split(',')})
    # read in akas data set
    print('Reading in akas...')
    #title_akas_df = pd.read_csv("title.akas.tsv", sep='\t', dtype=str)
    # read in basics data set
    print('Reading in basics...')
    #title_basics_df = pd.read_csv("title.basics.tsv", sep='\t', dtype=str)
    # read in ratings data set
    print('Reading in ratings...')
    #title_ratings_df = pd.read_csv("title.ratings.tsv", sep='\t')
    # remove all foreign entries from akas data set
    #title_akas_df = title_akas_df[title_akas_df['region'] == 'US']
    # remove all non-movie entries from basics data set
    #title_basics_df = title_basics_df[(title_basics_df['titleType'] == 'movie') | (title_basics_df['titleType'] == 'tvMovie')]
    #merge_databases(name_basics_df, title_akas_df, title_basics_df, title_ratings_df)
    print('Reading in Final...')
    final = pd.read_csv("final.tsv", sep='\t', dtype=str)
    print('Splitting the data...')
    split_attributes(final)


if __name__ == '__main__':
    main(sys.argv[1:])


