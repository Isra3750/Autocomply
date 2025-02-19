# First, import the required lib
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Step 1: import the two excel file - input file and reference file
df_main = pd.read_excel('Excel_file/Main.xlsx')
df_compare = pd.read_excel('Excel_file/Compare.xlsx')

# Import thai compatible model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encode all statements from Main.xlsx as a single batch
main_statements = df_main['Statement'].tolist()
main_embeddings = model.encode(main_statements, convert_to_tensor=True)

# Create similarity function - single batch variant
def find_match(statement, main_df, threshold=0.2):
    """
    Finds the best match for 'statement' within 'main_df' using sentence-transformers semantic similarity.

    Args:
        statement (str): The statement from df_compare to match against df_main.
        main_df (pd.DataFrame): DataFrame containing 'Statement' and 'Document' columns.
        threshold (float): Minimum similarity score required to consider a match valid.

    Returns:
        tuple: (best_document, best_statement, best_score)
            - best_document: The matching 'Document' from df_main
            - best_statement: The matched statement from df_main
            - best_score: The highest cosine similarity score found
    """
    # Encode the input statement once
    embedding_input = model.encode(statement, convert_to_tensor=True)

    # Compute similarity to all main embeddings at once (shape: (1, n_main))
    similarity_scores = util.pytorch_cos_sim(embedding_input, main_embeddings)[0]
    # Get the best score and its index
    best_score, best_idx = similarity_scores.max(dim=0)
    best_score = best_score.item()
    best_idx = best_idx.item()

    # Get the best match
    if best_score >= threshold:
        # Retrieve the corresponding row from df_main
        best_document = main_df.iloc[best_idx]['Document']  # Adjust column name if needed
        best_statement = main_df.iloc[best_idx]['Statement']
        return best_document, best_statement, best_score
    else:
        return None, None, best_score

# Step 2: Loop through each row of the input DataFrame
Result = []
for _ , row in df_compare.iterrows():
    # Use function to find the best match
    document, statement, score = find_match(row['Statement'], df_main, threshold=0.2)

    # If not none, then append to result
    if document is not None:
        Result.append({
            'Number': row['Number'],
            'Statement': row['Statement'],
            'Matched Statement': statement,
            'Matched Document Reference': document,
            'Similarity Score': score
        })

# Step 3: Output the resulting DataFrame
output_df = pd.DataFrame(Result)
print(output_df)

# Step 4: Save to result excel output file
#output_df.to_excel('Excel_file/Result.xlsx', index=False)