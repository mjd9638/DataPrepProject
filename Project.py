"""
@Authors: Ty Minnick, Mike Dortz, Maham Imtiaz, Deryke Tang
Description: This file contains code for the semester project.
Date: 9/9/2021
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time


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
    final_df.to_csv('final.tsv', sep='\t', index=False)


def split_attributes(dataset):
    """
    Splits attributes with multiple values into multiple lines.
    :param dataset: the merged dataset
    """
    attributes = ['types', 'attributes', 'genres', 'primaryProfession']
    for each in attributes:
        dataset[each] = dataset[each].str.split(',')
        dataset = dataset.explode(each)
    print(dataset.head())
    dataset.to_csv('final.tsv', sep='\t', index=False)


def phase_1():
    """
    Satisfies all requirements for phase 1
    """
    # read in names data set
    print('Reading in names...')
    name_basics_df = pd.read_csv("name.basics.tsv", sep='\t', dtype=str,
                                 converters={'knownForTitles': lambda x: x.split(',')})
    # read in akas data set
    print('Reading in akas...')
    title_akas_df = pd.read_csv("title.akas.tsv", sep='\t', dtype=str)
    # read in basics data set
    print('Reading in basics...')
    title_basics_df = pd.read_csv("title.basics.tsv", sep='\t', dtype=str)
    # read in ratings data set
    print('Reading in ratings...')
    title_ratings_df = pd.read_csv("title.ratings.tsv", sep='\t')
    # remove all foreign entries from akas data set
    title_akas_df = title_akas_df[title_akas_df['region'] == 'US']
    # remove all non-movie entries from basics data set
    title_basics_df = title_basics_df[
        (title_basics_df['titleType'] == 'movie') | (title_basics_df['titleType'] == 'tvMovie')]
    merge_databases(name_basics_df, title_akas_df, title_basics_df, title_ratings_df)
    final = pd.read_csv("final.tsv", sep='\t', dtype=str)
    print('Splitting the data...')
    split_attributes(final)


def phase_2(final):
    """
    This function replaces all \\N values in the dataframe with None.
    Counts the total number of entries and number of unique entries in each column
    Gets the max and min of each numerical attribute
    :param final: final tsv file as dataframe
    """
    print("...replacing \\N values...")
    final.replace({"\\N": None}, inplace=True)
    # print("Counting num of total entries...")
    # print(final.count(axis=0))
    # print("....printing final head....")
    # print(final.head())

    # print("Number of unique entries: ")
    # print(final.nunique())

    pd.set_option('display.max_columns', None)
    print(final.head())
    print("getting min and max values for each column...")
    list_num_cols = ['deathYear', 'birthYear', 'ordering', 'numVotes', 'averageRating', 'runtimeMinutes', 'endYear',
                     'startYear']
    for col in list_num_cols:
        final[col] = pd.to_numeric(final[col], errors='coerce')
        final[col].dropna(inplace=True)
        print("min of ", col, ": ", final[col].min())
        print("max of ", col, ": ", final[col].max())

    # print("getting freq dist of isAdult")
    # print(final['isAdult'].value_counts())


def phase_3(df):
    """
    Satisfies all requirements for phase 3
    :param df: the dataframe with all the movie data in it
    """
    print("getting mean, median, mode, min and max values for each column...")
    # list of all columns with numeric values
    list_num_cols = ['deathYear', 'birthYear', 'ordering', 'numVotes', 'averageRating', 'runtimeMinutes', 'endYear',
                     'startYear']
    # for each column with numeric values
    for col in list_num_cols:
        # change column's value to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # drops all na values in the columns
        df[col].dropna(inplace=True)
        # call mean, median, mode, min and max values of each column
        print("mean of " + str(col) + ": ", df[col].mean())
        print("median of " + str(col) + ": ", df[col].median())
        print("mode of " + str(col) + ": ", df[col].mode())
        print("min of " + str(col) + ": ", df[col].min())
        print("max of " + str(col) + ": ", df[col].max())
        print()
        # display box plot of the column
        sns.boxplot(df[col])
        plt.show()
    # print the outliers
    print(df[df['deathYear'] < 750]["deathYear"].unique())
    print(df[df['birthYear'] < 750]["birthYear"].unique())
    print(df[df['ordering'] > 100]["ordering"])
    print(df[df['numVotes'] > 1800000]["numVotes"])
    print(df[df['runtimeMinutes'] > 10000]["runtimeMinutes"].unique())
    # graph non-numerics
    non_numeric_graphs(df)
    # graph remaining numerics
    df['averageRating'].plot(kind='hist', title='averageRating', xlabel='Rating', bins=10)
    plt.xlabel('Rating')
    plt.ylabel('Frequency (Millions)')
    plt.xticks([x for x in range(11)])
    plt.show()
    df[df['startYear'] != r"\N"]['startYear'].value_counts().sort_index().plot(kind='line', title='startYear', xlabel='Year', ylabel='Count (Millions)')
    plt.show()
    bivariates(df)


def non_numeric_graphs(df):
    """
    Generates and display histograms for non-numeric columns
    :param df:
    :return:
    """
    cols = ['titleType', 'isAdult', 'language', 'types', 'isOriginalTitle']
    for col in cols:
        print(df[col].value_counts())
    df['titleType'].value_counts().plot(kind='bar', title='titleType', xlabel='Title', ylabel='Count (Tens of Millions)')
    plt.xticks(rotation=0)
    plt.show()
    df['isAdult'].value_counts().plot.pie(title='isAdult', ylabel='', autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
    df[df['language'] != r"\N"]['language'].value_counts().plot(kind='bar', title='language', xlabel='Language', ylabel='Count')
    plt.xticks(rotation=0)
    plt.show()
    df[df['types'] != r"\N"]['types'].value_counts().plot(kind='bar', title='types', xlabel='Type', ylabel='Count (Tens of Millions)')
    plt.xticks(rotation=70)
    plt.show()
    df[df['isOriginalTitle'] != r"\N"]['isOriginalTitle'].value_counts().plot.pie(title='isOriginalTitle', ylabel='', autopct='%1.3f%%')
    plt.show()


def bivariates(df):
    """
    Generates and displays bivariate graphs.
    :param df: the merged dataset
    """
    df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
    df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce')
    df['isAdult'] = pd.to_numeric(df['isAdult'], errors='coerce')
    df['startYear'] = pd.to_numeric(df['isAdult'], errors='coerce')
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
    df.plot.scatter(x='averageRating', y='numVotes')
    plt.title("Average Rating vs. Number of Votes")
    plt.ylabel("numVotes (millions)")
    plt.show()
    pd.crosstab(df[df['language'] != r"\N"]['language'], df['titleType']).plot.bar()
    plt.xticks(rotation=0)
    plt.title("Title type vs. Language")
    plt.ylabel("Count")
    plt.show()
    df.plot(x='numVotes', y='runtimeMinutes', kind='scatter', color='orange')
    plt.title("Run Time vs. Number of Votes")
    plt.xlabel("numVotes (millions)")
    plt.show()
    pd.crosstab(df['genres'], df['isAdult']).plot.bar()
    plt.title("Adult title vs. Genre")
    plt.ylabel("Count (millions)")
    plt.show()


def main(argv):
    """ Runs the program """
    pd.set_option('display.max_rows', None)
    print('Reading in Final...')
    start = time.time()
    final = pd.read_csv("final.tsv", sep='\t', dtype=str)
    end = time.time()
    print("TIME:", end - start)

    # Phase 1:
    #phase_1()

    # Phase 2:
    #phase_2(final)

    # Phase 3:
    #phase_3(final)


if __name__ == '__main__':
    main(sys.argv[1:])

