# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",
#   "python-dotenv"
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json

from dotenv import load_dotenv
load_dotenv()

# Function to analyze the data (basic summary stats, missing values, correlation matrix)
def data_analysis(dataframe):
    print("Analyzing the data...")  # Debugging line
    # Summary statistics for numerical columns
    stats_summary = dataframe.describe()

    # Check for missing values
    missing_values = dataframe.isnull().sum()

    # Select only numeric columns for correlation matrix
    numeric_dataframe = dataframe.select_dtypes(include=[np.number])

    # Correlation matrix for numerical columns
    correlation_mtx = numeric_dataframe.corr() if not numeric_dataframe.empty else pd.DataFrame()

    print("Data analysis complete.")  # Debugging line
    return stats_summary, missing_values, correlation_mtx


# Function to generate visualizations (correlation heatmap, outliers plot, and distribution plot)
def visualize_data(correlation_mtx, dataframe, output_dir):
    print("Generating visualizations...")  # Debugging line
    # Generate a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_mtx, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(heatmap_file)
    plt.close()
    print("Visualizations generated.")  # Debugging line
    return heatmap_file 



# Function to generate a detailed story using the new OpenAI API through the proxy
def story_llm(prompt, context, max_tokens=1000):
    print("Generating story using LLM...")  # Debugging line
    try:
        # Get the AIPROXY_TOKEN from the environment variable
        token = os.getenv("AIPROXY_TOKEN") # os.environ["AIPROXY_TOKEN"]

        # Set the custom API base URL for the proxy
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        # Construct the full prompt
        full_prompt = f"""
        Based on the following data analysis, please generate a creative and engaging story. The story should include multiple paragraphs, a clear structure with an introduction, body, and conclusion, and should feel like a well-rounded narrative.

        Context:
        {context}

        Data Analysis Prompt:
        {prompt}

        The story should be elaborate and cover the following:
        - An introduction to set the context.
        - A detailed body that expands on the data points and explores their significance.
        - A conclusion that wraps up the analysis and presents any potential outcomes or lessons.
        - Use transitions to connect ideas and keep the narrative flowing smoothly.
        - Format the story with clear paragraphs and structure.
        """

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # Prepare the body with the model and prompt
        data = {
            "model": "gpt-4o-mini",  # Specific model for proxy
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        # Send the POST request to the proxy
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check for successful response
        if response.status_code == 200:
            # Extract the story from the response
            story = response.json()['choices'][0]['message']['content'].strip()
            print("Story generated.")  # Debugging line
            return story
        else:
            print(f"Error with request: {response.status_code} - {response.text}")
            return "Failed to generate story."

    except Exception as e:
        print(f"Error: {e}")
        return "Failed to generate story."

def create_readme(ai_story):
    """
    Generate a README file that includes enhanced documentation and visualizations.
    """
    with open('README.md', 'w', encoding='utf-16') as readme_file:
        readme_file.write(ai_story + "\n\n")

# Main function that integrates all the steps
def main(csv_file):
    print("Starting the analysis...")  # Debugging 
    
    # output_folder = os.getcwd()

    # Try reading the CSV file with 'ISO-8859-1' encoding to handle special characters
    try:
        dataframe = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully!")  # Debugging line
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        return

    stats_summary, missing_values, correlation_mtx = data_analysis(dataframe)
    # outliers = detect_outliers(dataframe)

    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the data
    heatmap_file = visualize_data(correlation_mtx, dataframe, output_dir)

    # Limit the context size for the LLM to ensure it fits within 1000 tokens
    summary_of_context = f"""
    Dataset Analysis:
    Summary Statistics (First 5 columns):
    {stats_summary.iloc[:, :5]}

    Missing Values:
    {missing_values.head()}

    Correlation Matrix (First 5 columns):
    {correlation_mtx.iloc[:, :5]}

    """

    # Generate the story using the LLM
    story = story_llm("Generate a nice and creative story from the analysis", 
                         context=summary_of_context, 
                         max_tokens=1000)

    # print(f"Generated Story:\n{story}")

    create_readme(story)
        
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])