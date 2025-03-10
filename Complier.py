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

# Batch encoding
print("Encoding compare statements in batch...")
compare_statements = df_compare['Statement'].tolist()
compare_embeddings = model.encode(compare_statements, convert_to_tensor=True, show_progress_bar=True)

print("Computing similarity scores in a single pass...")
# This creates a similarity matrix of shape (len(df_compare), len(df_main))
similarity_matrix = util.pytorch_cos_sim(compare_embeddings, main_embeddings)

# Find the highest similarity score for each row in df_compare
best_scores, best_idxs = similarity_matrix.max(dim=1) # find highest similarity score in each row, dim = 1 means across column

# Set a threshold if you want to discard matches below a certain score
threshold = 0.1
Result = []

# Loop through each compare statement once, retrieving the best match
for i in range(len(compare_statements)):
    score = best_scores[i].item()
    idx = best_idxs[i].item()
    if score >= threshold:
        best_document = df_main.iloc[idx]['Document']
        best_statement = df_main.iloc[idx]['Statement']
        folder_location = df_main.iloc[idx]['Folder location']
        Result.append({
            'Number': df_compare.iloc[i]['Number'],
            'Statement': compare_statements[i],
            'Matched Statement': best_statement,
            'Matched Document Reference': best_document,
            'Similarity Score': score,
            'Folder location': folder_location
        })

# Print the Dataframe result
output_df = pd.DataFrame(Result)
print(output_df)

# Use pandas to create an Excel file with XlsxWriter module with similarity coloring base on three conditions
output_file = 'Excel_file/Result.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # write a dataframe into an excel file
    output_df.to_excel(writer, sheet_name='Sheet1', index=False)

    # Get the workbook and worksheet object
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
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

    # Set column width for each column
    worksheet.set_column('A:A', 8)
    worksheet.set_column('B:B', 50)
    worksheet.set_column('C:C', 50)
    worksheet.set_column('D:D', 65)
    worksheet.set_column('F:F', 30)
    
    # column E format is set to percentage instead, round to 2 decimal point
    percentage_format = workbook.add_format({'num_format': '0.00%'})
    worksheet.set_column('E:E', 14, percentage_format)


# Get end time and total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time:.3f} seconds taken")
