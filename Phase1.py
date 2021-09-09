"""
@Authors: Ty Minnick. Mike Dortz, Maham Imtiaz, Deryke Tang
Description: This file contains preliminary data mining for phase 1 of the project.
Date: 9/9/2021
"""

import sys
import pandas


def main(argv):
    # name_basics_df = pandas.read_csv("name.basics.tsv", sep='\t')
    # title_akas_df = pandas.read_csv("title.akas.tsv", sep='\t', dtype={'titleId': 'str','ordering': 'int',
    #                                                                    'title': 'str', 'region': 'str',
    #                                                                    'language': 'str', 'types': 'str',
    #                                                                    'attributes': 'str', 'isOriginalTitle': 'str'})
    # title_basics_df = pandas.read_csv("title.basics.tsv", sep='\t', dtype={'tconst': 'str', 'titleType': 'str',
    #                                                                        'primaryTitle': 'str',
    #                                                                        'originalTitle': 'str', 'isAdult': 'str',
    #                                                                        'startYear': 'str', 'endYear': 'str',
    #                                                                        'runtimeMinutes': 'str', 'genres': 'str'})
    title_ratings_df = pandas.read_csv("title.ratings.tsv", sep='\t')


if __name__ == '__main__':
    main(sys.argv[1:])


