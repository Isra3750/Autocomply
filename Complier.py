# Import the required lib
import pandas as pd # for data manipulation

# Sentence Transformers enables the transformation of sentences into vector spaces
from sentence_transformers import SentenceTransformer, util # util provides helper function for embeddings such as the function pytorch_cos_sim to compute cosine similarity
from tqdm import tqdm # for progress bar
import time # for total time
import pickle # for caching main embeddings
import os

# Import the two excel file - input file and reference file
df_main = pd.read_excel('Excel_file/Main.xlsx')
df_compare = pd.read_excel('Excel_file/Compare.xlsx')

# Record start time
start_time = time.time()

# Import thai compatible model
print("Start loading model...")
with tqdm(total=1, desc="Loading Model") as pbar:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    pbar.update(1)
print("Model loaded!")

# Cache file for main embeddings
cache_file = 'main_embeddings.pkl'

# Use pickle to cache main file embeddings - load if already created, create if not
# Note: When excel file is changed, the embeddings will need to be re-created, delete the cache file (main_embeddings.pkl)
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        main_embeddings = pickle.load(f)
    print("Loaded cache file for main embeddings!")
else:
    # Encode all statements from Main.xlsx as a single batch
    print("Start embedding main statements...")
    main_statements = df_main['Statement'].tolist()
    main_embeddings = model.encode(main_statements, convert_to_tensor=True, show_progress_bar=True)

    # Cache the embeddings
    with open(cache_file, 'wb') as f:
        pickle.dump(main_embeddings, f)
    
    print("Created cache file for main embeddings!")

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
    similarity_scores = util.pytorch_cos_sim(embedding_input, main_embeddings)[0] # [0] extracts the first row from 2D tensor, this is similarity score for each statement in main_df
    # Get the best score and its index
    best_score, best_idx = similarity_scores.max(dim=0) # This will get the highest similarity score from the 1D tensor and its index, dim = 0 means row
    best_score = best_score.item() # convert pytorch into float, Ex. 0.95
    best_idx = best_idx.item() # convert pytorch into float, Ex. 3

    # Threshold checking, return none if below threshold, which will skipped the append later
    if best_score >= threshold:
        # Retrieve the corresponding row from df_main with the best index from before
        best_document = main_df.iloc[best_idx]['Document']  # Adjust column name if needed
        best_statement = main_df.iloc[best_idx]['Statement']
        folder_location = main_df.iloc[best_idx]['Folder location']
        return best_document, best_statement, best_score, folder_location
    else:
        return None, None, best_score, None

Result = []
# Loop through each row, tqdm for progress bar
for _, row in tqdm(df_compare.iterrows(), total=len(df_compare), desc="Processing rows"):
    # Use function to find the best match
    document, statement, score, location = find_match(row['Statement'], df_main, threshold=0.2)

    # If not none, then append to result
    if document is not None:
        Result.append({
            'Number': row['Number'],
            'Statement': row['Statement'],
            'Matched Statement': statement,
            'Matched Document Reference': document,
            'Similarity Score': score,
            'Folder location': location
        })

# Print the Dataframe result
output_df = pd.DataFrame(Result)
print(output_df)

# Save to result excel output file, with coloring for similarity score
output_file = 'Excel_file/Result.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    output_df.to_excel(writer, sheet_name='Sheet1', index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # Format the "Similarity Score" column (assumed to be column E) to display as percentage
    percentage_format = workbook.add_format({'num_format': '0.00%'})
    worksheet.set_column('E:E', 18, percentage_format)
    
    # Determine the cell range for the Similarity Score column (row 2 to the last row)
    num_rows = len(output_df)
    cell_range = f'E2:E{num_rows + 1}'
    
    # Apply conditional formatting:
    # Red for scores below 80% (< 0.8)
    red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    worksheet.conditional_format(cell_range, {
        'type': 'cell',
        'criteria': '<',
        'value': 0.8,
        'format': red_format
    })
    
    # Orange for scores between 80% and 95% (0.8 to 0.95)
    orange_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
    worksheet.conditional_format(cell_range, {
        'type': 'cell',
        'criteria': 'between',
        'minimum': 0.8,
        'maximum': 0.95,
        'format': orange_format
    })
    
    # Green for scores 95% and above (>= 0.95)
    green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    worksheet.conditional_format(cell_range, {
        'type': 'cell',
        'criteria': '>=',
        'value': 0.95,
        'format': green_format
    })

# Get end time and total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time:.3f} seconds taken")
