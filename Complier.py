# First, import the required lib
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Step 1: import the two excel file - input file and reference file
df_main = pd.read_excel('Excel_file/Main.xlsx')
df_compare = pd.read_excel('Excel_file/Compare.xlsx')

# Create spacy nlp object
# load en_core_web_md (small model), en_core_web_lg (large model), en_core_web_trf (largest)
# pip uninstall en-core-web-lg
#nlp = spacy.load("en_core_web_lg")

# Import thai compatable model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Create similarity function
def find_match(statement, main_df, threshold=0.2):
    """
    Finds the best match for 'statement' within 'main_df' using sentence-transformers semantic similarity.

    Args:
        statement: str
        main_df: xlsx file
    Returns:
        tuple (document reference, matched statement, similarity score)
    """
    # Encode the input statement once
    embedding_input = model.encode(statement, convert_to_tensor=True)

    # Find Best Match -> Score (debugging), Document (actual use), Statement (debugging)
    best_score = 0
    best_document = None
    best_statement = None
    for _, row in main_df.iterrows(): # iterrow output a tuple (index, row)
        # Encode each statement once
        embedding_main = model.encode(row['Statement'], convert_to_tensor=True)
        
        # Calculate cos similarity
        score_tensor = util.pytorch_cos_sim(embedding_input, embedding_main)
        score = score_tensor.item()

        # Update best match
        if score > best_score:
            best_score = score
            best_document = row['Document']
            best_statement = row['Statement']
    if best_score >= threshold:
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
output_df.to_excel('Excel_file/Result.xlsx', index=False)