import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_excel('your_file.xlsx')

def format_prompt(comment):
    return f"Comment: \"{comment}\"\nQuestion: Is this comment genuine or trolling? Explain your reasoning."


train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
