# Automacally highlights most represented words in pdf files.

Add as many pdf files as desired in './input/' then run the script `python3 highlight_pdf.py` and find highlighted files in './output/'.

Script parameters are:
- `-c` to control threshold occurrence,
- `-p` to control maximum number of words to keep.
- `-i` to control input folder,
- `-o` to control output folder,
- `-f` to set a specific file name (otherwise all files in the input folder are processed)
- `-b` if True, export to same folder as input, and replace original (create .bkp file)
- `-r` if True, restore original file from .bkp file and delete .bkp file
- `-a` if True, remove all annotations from the pdf

Use `python3 highlight_pdf.py --help` to show full help.


(pdf source : https://dagrs.berkeley.edu/sites/default/files/2020-01/sample.pdf)
