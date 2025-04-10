import os
import pandas as pd
import textwrap
import google.generativeai as genai

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Retrieve the API key from an environment variable.
# Make sure to set 'GOOGLE_API_KEY' in your system or Visual Studio's run configuration.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDSTs3Ys4RKpejH2xLia8kwQN_NkkIlnyU')
genai.configure(api_key=GOOGLE_API_KEY)

# List available models that support content generation.
model_name = None
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        model_name = m.name
        break

if model_name is None:
    raise ValueError("No suitable model found!")
else:
    print("Using model:", model_name)

# --- Load Data from CSV ---
# Assume your CSV file is named 'data.csv' and has two columns:
# Column 0: content (中文文本)
# Column 1: label (0 or 1)
df = pd.read_csv("/Users/yui/Downloads/NLP_FP/NLP_FP.csv", header=None)
df.columns = ['content', 'label']

# --- Define Chinese Prompt Template ---
prompt_template = (
    "請根據以下文本判斷其是否為反串。如果是，請回答 '1'；如果不是，請回答 '0'。\n"
    "文本如下：\n\n"
)

# --- Send Each Text to Gemini and Collect Predictions ---
predictions = []
for idx, row in df.iterrows():
    text = row['content']
    prompt = prompt_template + text
    # print(f"Sending prompt for text [{idx}]: {text}\n")
    
    # Use generate_text instead of generate
    response = genai.chat(
        model=model_name,
        prompt=prompt,
        temperature=0.0,         # Using low temperature for deterministic output
        max_output_tokens=5      # Adjust as needed
    )
    
    # Extract the answer from the response (assumed to be in the result property)
    answer = response.result.strip()  
    predictions.append(answer)
    
    print(f"Prediction for text [{idx}]: {answer}\n{'-'*40}\n")

# --- Optionally, Add Predictions to Your DataFrame ---
df['prediction'] = predictions

# You can then review or save the results:
print(df.head())
df.to_csv("data_with_predictions.csv", index=False)
