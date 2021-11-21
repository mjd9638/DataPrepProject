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
    df[df['startYear'] != r"\N"]['startYear'].value_counts().sort_index().plot(kind='line', title='startYear',
                                                                               xlabel='Year', ylabel='Count (Millions)')
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
    df['titleType'].value_counts().plot(kind='bar', title='titleType', xlabel='Title',
                                        ylabel='Count (Tens of Millions)')
    plt.xticks(rotation=0)
    plt.show()
    df['isAdult'].value_counts().plot.pie(title='isAdult', ylabel='', autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
    df[df['language'] != r"\N"]['language'].value_counts().plot(kind='bar', title='language', xlabel='Language',
                                                                ylabel='Count')
    plt.xticks(rotation=0)
    plt.show()
    df[df['types'] != r"\N"]['types'].value_counts().plot(kind='bar', title='types', xlabel='Type',
                                                          ylabel='Count (Tens of Millions)')
    plt.xticks(rotation=70)
    plt.show()
    df[df['isOriginalTitle'] != r"\N"]['isOriginalTitle'].value_counts().plot.pie(title='isOriginalTitle', ylabel='',
                                                                                  autopct='%1.3f%%')
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


def  phase_4_remove_missing(final):
    """
    TODO: endyear: remove column
    TODO: runtime_minutes: replace nan with median
    TODO: genres: replace nan with Unknown
    averageRating: replace with median
    numVotes: replace with median (rounded)
    ordering: remove column
    title: replace with Unknown
    region: remove record
    language: replace with Unknown
    types: replace with imdbDisplay
    attributes: replace with None
    isOriginalTitle: no missing values after removing non-US values (?)
    nconst: no blanks
    birthyear: replace with -1
    deathyear: replace with -1
    primaryprofession: unknown
    """
    # print('getting only US values')
    # final = final[final['region'].notna()]
    # print('writing to file')
    # final.to_csv('phase_4_final.tsv', sep='\t', index=False)

    # remove COLUMNS = endYear, ordering
    final.drop(['endYear', 'ordering'], axis=1, inplace=True)

    # replace NA with Unknown for the following columns
    unknown_nan_attr = ['genres', 'title', 'language', 'primaryProfession']
    for attr in unknown_nan_attr:
        final[attr].fillna('Unknown', inplace=True)

    median_runTimeMinutes = 100
    final['runtimeMinutes'].fillna(median_runTimeMinutes, inplace=True)

    median_avgRating = 6.4
    final['averageRating'].fillna(median_avgRating, inplace=True)

    final['numVotes'].fillna(3730, inplace=True)

    final['types'].fillna('imdbDisplay', inplace=True)

    final['attributes'].fillna('None', inplace=True)

    final['birthYear'].fillna(-1, inplace=True)

    final['deathYear'].fillna(-1, inplace=True)

    final.to_csv('phase_4_final_new.tsv', sep='\t', index=False)


def phase_4_normalize(final):
    """
    Normalize the following:
    runtimeMinutes
    AverageRating
    numVotes
    """

    print('converting runtime minutes to int')
    final['runtimeMinutes'] = final['runtimeMinutes'].astype(int)

    print('normalizing runtime minutes')
    runtime_min_mean = final['runtimeMinutes'].mean()
    runtime_min_std = final['runtimeMinutes'].std()
    # mean normalization of runtimeMinutes
    final['runtimeMinutes'] = (final['runtimeMinutes'] - runtime_min_mean) / runtime_min_std

    print('converting avg rating to float')
    final['averageRating'] = final['averageRating'].astype(float)

    print('normalizing avg rating')
    averageRating_mean = final['averageRating'].mean()
    averageRating_std = final['averageRating'].std()

    # mean normalization of AverageRating
    final['averageRating'] = (final['averageRating'] - averageRating_mean) / averageRating_std

    print('converting numVotes to float')
    final['numVotes'] = final['numVotes'].astype(float)

    print('normalizing numVotes')
    # mean normalization of numVotes
    numVotes_mean = final['numVotes'].mean()
    numVotes_std = final['numVotes'].std()

    final['numVotes'] = (final['numVotes'] - numVotes_mean) / numVotes_std

    print('writing normalized data to file')
    final.to_csv('phase_4_final_normalized.tsv', sep='\t', index=False)


def phase_4_remove_invalid(final):
    """
    remove invalid year values from teh following:
    startYear
    birthYear
    deathYear
    """

    print('removing invalid start years')
    final = final[(final['startYear'].str.len() == 4) | (final['startYear'] == -1)]

    print('removing invalid birth years')
    final = final[(final['birthYear'].str.len() == 4) | (final['birthYear'] == -1)]

    print('removing invalid death years')
    final = final[(final['deathYear'].str.len() == 4) | (final['deathYear'] == -1)]

    print('writing final to file')
    final.to_csv('phase_4_rem_invalid.tsv', sep='\t', index=False)


def main(argv):
    """ Runs the program """
    pd.set_option('display.max_rows', None)
    print('Reading in Final...')
    start = time.time()
    # final = pd.read_csv("final.tsv", sep='\t', dtype=str, na_values='\\N')

    # reading final for removing missing/ empty values
    # final = pd.read_csv("phase_4_final.tsv", sep='\t', dtype=str, na_values='\\N')

    # reading dataframe with missing values removed, to remove invalid
    # final = pd.read_csv("phase_4_final_new.tsv", sep='\t', dtype=str, na_values='\\N')

    # reading normalized data to normalize
    final = pd.read_csv("phase_4_rem_invalid.tsv", sep='\t', dtype=str, na_values='\\N')

    end = time.time()
    print("TIME:", end - start)

    # Phase 1:
    # phase_1()

    # Phase 2:
    # phase_2(final)

    # Phase 3:
    # phase_3(final)

    # Phase 4:
    # phase_4_remove_missing(final)

    # phase_4_remove_invalid(final)

    phase_4_normalize(final)


if __name__ == '__main__':
    main(sys.argv[1:])
