import pandas as pd  # Import pandas, a library used for data manipulation and analysis.
import re  # Import the re module for regular expressions, which is useful for text cleaning.

# Read the CSV file into a pandas DataFrame.
# 'TaylorSwift.csv' is expected to have a column named 'Lyric' that contains the song lyrics.
df = pd.read_csv('TaylorSwift.csv')

# Function to clean lyrics by removing unwanted characters.
def clean_lyrics(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove all non-alphabetic characters and punctuation (keep only letters and spaces).
    text = text.lower()  # Convert all text to lowercase to normalize it.
    return text  # Return the cleaned text.

# Create a new column in the DataFrame to store cleaned lyrics.
# Convert the 'Lyric' column to string type and apply the `clean_lyrics` function to each entry.
df['Cleaned_Lyric'] = df['Lyric'].astype(str).apply(clean_lyrics)

# Initialize an empty string to concatenate all the cleaned lyrics.
lyrics = ""
for i in df["Cleaned_Lyric"]:
    lyrics += str(i)  # Add each cleaned lyric to the `lyrics` string.
    lyrics += "\n \n"  # Add a double newline character between each lyric for separation.

# Write the cleaned and concatenated lyrics to a text file named 'lyrics.txt'.
with open('lyrics.txt', 'w') as f:
    f.write(lyrics)