# First, import the required lib
import pandas as pd
import spacy

# Step 1: import the two excel file - input file and reference file
df_main = pd.read_excel('Excel_file/Main.xlsx')
df_compare = pd.read_excel('Excel_file/Compare.xlsx')

# Create spacy nlp object -> load en_core_web_md (small model), en_core_web_lg (large model)
nlp = spacy.load("en_core_web_trf")

# Create similarity function
def find_match(statement, main_df, threshold = 0):
    """
    Find if statement string is located in main_df and return statement else return none
    Args:
        statement: str
        main_df: xlsx file
    Returns:
        tuple (document reference, matched statement, similarity score)
    """
    # Process the input statement into a spacy doc
    doc_input = nlp(statement)

    # Find Best Match -> Score (debugging), Document (actual use), Statement (debugging)
    best_score = 0
    best_document = None
    best_statement = None
    for _, row in main_df.iterrows(): # iterrow output a tuple (index, row)
        # Process the statement into a spacy doc
        doc_main = nlp(row['Statement'])
        # Calculate similarity
        score = doc_input.similarity(doc_main)
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
    document, statement, score = find_match(row['Statement'], df_main)

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