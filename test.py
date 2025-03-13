# Import the required lib
import pandas as pd # for data manipulation

# Sentence Transformers enables the transformation of sentences into vector spaces
from sentence_transformers import SentenceTransformer, util # util provides helper function for embeddings such as the function pytorch_cos_sim to compute cosine similarity
from tqdm import tqdm # for progress bar
import time # for total time
import pickle # for caching main embeddings
import os
import fitz
import shutil

# =====
# Section 1: Embeddings excel file
# =====

df_main = pd.read_excel('Excel_file/Mapped_PDF.xlsx')
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

# =====
# Section 2: Compute similarity
# =====

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
        best_PDF = df_main.iloc[idx]['PDF_name']
        folder_location = df_main.iloc[idx]['Folder location']
        Result.append({
            'Number': df_compare.iloc[i]['Number'],
            'Statement': compare_statements[i],
            'Matched Statement': best_statement,
            'Matched Document Reference': best_document,
            'Similarity Score': score,
            'Related PDF': best_PDF,
            'Folder location': folder_location
        })

# Print the Dataframe result
output_df = pd.DataFrame(Result)
print(output_df)

# =====
# Section 3: Create and format Excel file
# =====

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
    worksheet.set_column('F:F', 13)
    worksheet.set_column('G:G', 22)
    
    # column E format is set to percentage instead, round to 2 decimal point
    percentage_format = workbook.add_format({'num_format': '0.00%'})
    worksheet.set_column('E:E', 14, percentage_format)

# =====
# Section 4: Find related PDF and adjust annotations
# =====

# Clear the output folder so that only the new PDFs will be there
output_dir = 'output'
if os.path.exists(output_dir):
    print("Clearing output folder...")
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print("Unlinking file: " + file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print("Removing directory: " + file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    print("Making output folder...")
    os.makedirs(output_dir)

print("\nUpdating PDF annotations based on Result DataFrame...")
"""
# Loop over each row in the result DataFrame
for index, row in output_df.iterrows():
    # Skip rows with low similarity scores -> prevent excessive processing
    if row['Similarity Score'] < 0.8:
        print(f"Skipping PDF for row {index} due to low similarity score: {row['Similarity Score']:.3f}")
        continue
    
    # Retrieve the PDF file name from the 'Related PDF' column
    pdf_name = row['Related PDF']
    # Use Numbers row as the name and content
    annotation_content = str(row['Number'])
    
    # Build the PDF file path (assuming the name does not include the .pdf extension)
    pdf_path = f'document/{pdf_name}.pdf' if not pdf_name.lower().endswith('.pdf') else f'document/{pdf_name}'
    print(f"Opening PDF: {pdf_path} ...")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open {pdf_path}: {e}")
        continue
    
    # Loop over all pages and update free-text annotations
    for page in doc:
        annots = page.annots()
        if annots:
            for annot in annots:
                if "FreeText" in annot.type:
                    annot.set_info(content=annotation_content, title="Oracle")
                    annot.update()
                    print(f"Updated annotation in {pdf_name} to: {annotation_content}")
    
    # Save the updated PDF to the output folder with a unique name
    output_pdf_path = os.path.join(output_dir, f'{annotation_content}.pdf')
    doc.save(output_pdf_path)
    doc.close()
    print(f"Saved updated PDF to: {output_pdf_path}")
"""
# Get end time and total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time:.3f} seconds taken")