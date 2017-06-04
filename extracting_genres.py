import pickle
import os.path
from config import Config as conf
from data_utility import *
import re


def extract_base_dataset_genres(training_genres=False, validation_genres=False):
    if not training_genres and not validation_genres:
        return

    print("Loading metadata from path {}".format(conf.genres_basedataset_metainfo_path))
    metadata = open(conf.genres_basedataset_metainfo_path, 'r')
    genres = {}                     # moview ID -> lowercase genre
    unique_genres = set()
    for line in metadata:
        tokens = line.strip().split("\t")
        if len(tokens) < 33:
            continue
        # print(len(tokens))
        # print(tokens)
        id = tokens[0].strip()
        all_genres = tokens[6].strip().lower().split(",")
        first_genre = all_genres[0].strip()
        unique_genres.add(first_genre)
        genres[id] = first_genre

    t_genre_labels = None
    v_genre_labels = None

    if training_genres:
        t_genre_labels = extract_basedataset(conf.genres_basedataset_training_path, genres)
    if validation_genres:
        v_genre_labels = extract_basedataset(conf.genres_basedataset_validation_path, genres)

    return genres, unique_genres, t_genre_labels, v_genre_labels


def extract_basedataset(training_path, genres):
    scripts = open(training_path, 'r')

    genre_labels = []           # in the order of appearance of dialogues
    for line in scripts:
        tokens = line.strip().split("\t")
        id = tokens[0].strip()
        genre_labels.append(genres[id])

    return genre_labels


def extract_cornell_genres():
    metadata = open(conf.genres_cornell_metainfo_path, 'r', encoding="ISO-8859-1")
    unique_genres = set()
    movie2genre = {}
    for line in metadata:
        tokens = line.strip().split(' +++$+++ ')
        movie_id = tokens[0].strip()
        all_genres = tokens[5].strip()
        all_genres = all_genres[1:-1].split(",")
        first_genre = all_genres[0].strip()[1:-1]
        # if len(all_genres) == 0 or len(all_genres) == 1:
        #     print("For movie {} number of genres is {} and first genre is {}".format(movie_id, len(all_genres), first_genre))
        if first_genre == '':
            first_genre = 'genre'
        unique_genres.add(first_genre)
        movie2genre[movie_id] = first_genre

    return movie2genre, unique_genres


def mainFunc():
    matching, unique_genres, t_genre_labels, _ = extract_base_dataset_genres(True, False)
    print("Number of unique genres is: {}".format(len(unique_genres)))
    print(unique_genres)
    matching2, unique_genres2 = extract_cornell_genres()
    print("Number of unique genres is: {}".format(len(unique_genres2)))
    print(unique_genres2)
    its = unique_genres.intersection(unique_genres2)
    print(len(its))
    print(its)
    diff1 = unique_genres.difference(unique_genres2)
    diff2 = unique_genres2.difference(unique_genres)
    print("Genres in base dataset that are not in Cornell {}".format(diff1))
    print("Genres in cornell that are not in base dataset {}".format(diff2))

if __name__ == "__main__":
    mainFunc()


