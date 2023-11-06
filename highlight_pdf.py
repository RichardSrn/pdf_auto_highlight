import os
import re

import click
import fitz
import nltk
import numpy as np
from tqdm import tqdm


@click.command()
@click.option("--threshold_occurrence", "-c",
              default=5,
              help="Minimum number of occurrence of a word to be considered as relevant.")
@click.option("--threshold_podium", "-p",
              default=15,
              help="Number of most represented words to keep")
@click.option("--path_input", "-i",
                default="./input/",
                help="Path to the input folder.")
@click.option("--path_output", "-o",
                default="./output/",
                help="Path to the output folder.")
@click.option("--file_name", "-f",
                default=None,
                help="Name of the file to highlight. If None, all files in the input folder will be highlighted.")
@click.option("--backup_and_replace", "-b",
                is_flag=True,
                help="If True, export to same folder as input, and replace original (create .bkp file)")
@click.option("--restore_bkp", "-r",
                is_flag=True,
                help="If True, restore original file from .bkp file and delete .bkp file")
@click.option("--clean_annotations", "-a",
                is_flag=True,
                help="If True, remove all annotations from the pdf")
def main(
        threshold_occurrence=5,
        threshold_podium=15,
        path_input="./input/",
        path_output="./output/",
        file_name=None,
        backup_and_replace=False,  # if True, export to same folder as input, and replace original (create .bkp file)
        restore_bkp=False,  # if True, restore original file from .bkp file and delete .bkp file
        clean_annotations=False, # if True, remove all annotations from the pdf
):

    assert threshold_occurrence > 0,\
        "threshold_occurrence must be greater than 0"
    assert threshold_podium > 0,\
        "threshold_podium must be greater than 0"
    assert os.path.exists(path_input), \
        f"path_input {path_input} does not exist"
    assert os.path.exists(path_output), \
        f"path_output {path_output} does not exist"
    assert file_name is None or os.path.exists(os.path.join(path_input, f"{file_name}.pdf")), \
        f"file_name {file_name} does not exist in path_input {path_input}"
    assert not (restore_bkp and backup_and_replace), \
        "backup_and_replace and restore_bkp cannot be True at the same time"
    assert not (restore_bkp and clean_annotations), \
        "restore_bkp and clean_annotations cannot be True at the same time"


    print("*** PDF AUTO HIGHLIGHT ***")
    print(f"keep all works with more than {threshold_occurrence} occurrences")
    print(f"keep the {threshold_podium} most represented words")
    print(f"input folder :".ljust(16), f"{path_input}")
    print(f"output folder :".ljust(16), f"{path_output}")
    if file_name is None :
        print("file_name :".ljust(16), f"all files in input folder")
    else :
        print(f"file_name :".ljust(16), f"{file_name}")
    if backup_and_replace :
        print(f"file(s) will be replaced by the highlighted version and a .bkp file will be created")
    if restore_bkp :
        print(f"file(s) will be restored from the .bkp file")
    if clean_annotations :
        print(f"all annotations will be removed from the pdf")
    print("**************************", end="\n\n")


    if restore_bkp :
        # list all files in ./input/ and remove all files that are not pdf
        files = os.listdir(path_input)
        files = [f for f in files if f.endswith(".pdf.bkp")]
        for file_name in files:
            # remove .bkp from file_name
            file_name = file_name[:-8]
            # remove original file
            if os.path.exists(os.path.join(path_input, f"{file_name}.pdf")):
                os.remove(os.path.join(path_input, f"{file_name}.pdf"))
            # rename .bkp file to original file
            os.rename(os.path.join(path_input, f"{file_name}.pdf.bkp"),
                      os.path.join(path_input, f"{file_name}.pdf"))
    else :
        if file_name is None :
            # list all files in ./input/ and remove all files that are not pdf
            files = os.listdir(path_input)
            files = [f for f in files if f.endswith(".pdf")]
        else :
            files = [file_name]

        for file_name in files:
            if file_name.endswith(".pdf"):
                file_name = file_name[:-4]

            if clean_annotations :
                ### OPEN PDF
                with fitz.open(os.path.join(path_input, f"{file_name}.pdf")) as doc:
                    ### CLEAN ANNOTATIONS
                    for page in doc:
                        annot = page.first_annot
                        while annot:
                            next_annot = annot.next
                            page.delete_annot(annot)
                            annot = next_annot
                    ### OUTPUT
                    if backup_and_replace :
                        # save doc in file_name.pdf
                        doc.save(os.path.join(path_input, f"{file_name}.pdf_"))
                        # copy original file into file_name.pdf.bkp
                        os.rename(os.path.join(path_input, f"{file_name}.pdf"),
                                  os.path.join(path_input, f"{file_name}.pdf.bkp"))
                        # rename file_name.pdf_ into file_name.pdf
                        os.rename(os.path.join(path_input, f"{file_name}.pdf_"),
                                  os.path.join(path_input, f"{file_name}.pdf"))
                    else :
                        doc.save(os.path.join(path_output, f"{file_name}_cleaned.pdf"))
            else :
                ### OPEN PDF
                with fitz.open(os.path.join(path_input, f"{file_name}.pdf")) as doc:
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

                with fitz.open(os.path.join(path_input, f"{file_name}.pdf")) as doc:
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
                    if backup_and_replace :
                        # save doc in file_name.pdf
                        doc.save(os.path.join(path_input, f"{file_name}.pdf_"))
                        # copy original file into file_name.pdf.bkp
                        os.rename(os.path.join(path_input, f"{file_name}.pdf"),
                                  os.path.join(path_input, f"{file_name}.pdf.bkp"))
                        # rename file_name.pdf_ into file_name.pdf
                        os.rename(os.path.join(path_input, f"{file_name}.pdf_"),
                                  os.path.join(path_input, f"{file_name}.pdf"))
                    else :
                        doc.save(os.path.join(path_output, f"{file_name}_highlighted.pdf"))


if __name__ == "__main__":
    main()
