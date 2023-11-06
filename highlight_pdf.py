import os
import re

import click
import fitz
import nltk
import numpy as np
from tqdm import tqdm


@click.command()
@click.option("--threshold_occurrence", "-o",
              default=5,
              help="Minimum number of occurrence of a word to be considered as relevant.")
@click.option("--threshold_podium", "-p",
              default=15,
              help="Number of most represented words to keep")
def main(
        threshold_occurrence=5,
        threshold_podium=15
):
    # list all files in ./input/ and remove all files that are not pdf
    files = os.listdir("./input/")
    files = [f[:-4] for f in files if f[-4:] == ".pdf"]

    for file_name in files:
        ### OPEN PDF
        with fitz.open(f"./input/{file_name}.pdf") as doc:
            ### GET THE MOST REPRESENTED WORDS
            words = {}
            for page in doc:
                text = page.get_text()
                text = re.sub(r"\W|\d", " ", text)
                text = re.sub(r"[ ]{2,}", " ", text)

                text = text.lower()
                matches = re.split(r"\s|\(|\)|\n|\[|\]|\,|\.", text)

                for word in matches:
                    if word in words.keys():
                        words[word] += 1
                    else:
                        words[word] = 1

            ### REMOVE STOP WORDS

            ### KEEP ONLY MOST REPRESENTED WORDS
            # remove stop words from words
            nltk.download('stopwords', quiet=True)
            stop_words = nltk.corpus.stopwords.words('english')
            words = {key: value for key, value in words.items() if key not in stop_words}
            # remove keys with less than 3 characters
            words = {key: value for key, value in words.items() if len(key) > 3}
            # remove all keys with less than threshold_occurrence value in words
            words = {key: value for key, value in words.items() if value > threshold_occurrence}
            # remove duplicates (features and feature are considered duplicate for example)
            wds = list(words.keys())
            wds_to_drop = []
            for w in wds:
                if len([x for x in wds if w in x]) > 1:
                    duplicates = [x for x in wds if w in x]
                    # check if the difference between the length of the word and the length of the duplicates is less than 2
                    duplicates_length = [len(x) for x in duplicates]
                    # if max(duplicates_length) - min(duplicates_length) < 2:
                    #    # add all words in duplicates to wds_to_drop, except the shortest one
                    #    wds_to_drop += [x for x in duplicates if len(x) != min(duplicates_length)]
                    wds_to_drop += [x for x in duplicates if len(x) != min(duplicates_length)]
            words = {key: value for key, value in words.items() if key not in wds_to_drop}
            # keep only the threshold_podium most represented words
            words = dict(sorted(words.items(), key=lambda item: item[1], reverse=True)[:threshold_podium])

            most_represented_words = list(words.keys())

        # define random rgb colors with a for loop
        colors = np.random.rand(len(most_represented_words), 3)
        # avoid dark colors
        for i in range(colors.shape[0]):
            # check is the l2-norm of c is less than 0.75
            # if so, square_root c, until the l2-norm is greater than 0.75
            while np.linalg.norm(colors[i]) < 0.75:
                colors[i] = np.sqrt(colors[i])
            for j in range(colors.shape[1]):
                while colors[i, j] < 0.4:
                    colors[i, j] = np.sqrt(colors[i, j])
        colors = colors.round(2)

        i = 0
        print("Most represented words :")
        for word, occurrence in words.items():
            # print word with colorbox colors[i] and text color black in the terminal
            print(f"{str(occurrence).rjust(5)} occurrences : ", end="")
            print("(#rgb", end=" ")
            for c in tuple(colors[i].round(2)):
                print(f"{c}".ljust(4, "0"), end=" ")
            print(")", end=" ")
            print(":", end=" ")
            print(
                f"\033[48;2;{int(colors[i][0] * 255)};{int(colors[i][1] * 255)};{int(colors[i][2] * 255)}m\033[38;2;0;0;0m{word}\033[0m".ljust(
                    25))
            i += 1

        with fitz.open(f"./input/{file_name}.pdf") as doc:
            ### HIGHLIGHT
            i = 0
            for word in tqdm(most_represented_words):
                for page in doc:
                    text_instances = page.search_for(word)

                    for inst in text_instances:
                        # print(inst, type(inst))
                        highlight = page.add_highlight_annot(inst)
                        c = list(colors[i])
                        highlight.set_colors({"stroke": c})  # , "fill": (0.75, 0.8, 0.95)})
                        highlight.update()
                i += 1

            ### OUTPUT
            doc.save(f"./output/{file_name}_highlighted.pdf", garbage=4, deflate=True, clean=True)


if __name__ == "__main__":
    main()
