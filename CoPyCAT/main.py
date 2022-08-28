"""Main script - includes the ability to run CLI"""

from preprocessing.transcript_cleaner import create_dataframe

file_path = ".\example_transcripts\demo.txt"
print(create_dataframe(file_path))


if __name__ == '__main__':
    pass
else:
    pass