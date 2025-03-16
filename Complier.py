import os
import shutil
import pickle
import time
import pandas as pd
import fitz
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ---------------------------------------------------------------------
# Helper function: create or load embeddings for a given Mapped_PDF sheet
# ---------------------------------------------------------------------
def get_or_create_embeddings(df_main_sheet, model, sheet_name):
    """
    Given a DataFrame from Mapped_PDF.xlsx for a specific sheet (sheet_name),
    load or create cached embeddings for its 'Statement' column.
    Returns a list of embeddings in the same order as df_main_sheet.
    """
    os.makedirs('embeddings', exist_ok=True)  # ensure folder exists
    cache_file = f'embeddings/main_embeddings_{sheet_name}.pkl'
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            main_embeddings = pickle.load(f)
        print(f"Loaded cache file for sheet '{sheet_name}' embeddings!")
    else:
        print(f"Creating embeddings for sheet '{sheet_name}'...")
        main_statements = df_main_sheet['Statement'].tolist()
        main_embeddings = model.encode(main_statements, convert_to_tensor=True, show_progress_bar=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(main_embeddings, f)
        print(f"Created cache file for sheet '{sheet_name}' embeddings!")
    
    return main_embeddings

# ---------------------------------------------------------------------
# Helper function: process one sheet (ODB or OMI)
# ---------------------------------------------------------------------
def process_sheet(sheet_name, df_main, df_compare, model, writer):
    """
    - df_main: DataFrame of the current sheet from Mapped_PDF.xlsx
    - df_compare: DataFrame of the current sheet from Compare.xlsx
    - model: SentenceTransformer model
    - writer: ExcelWriter for the final Result.xlsx

    Returns a DataFrame of the results for this sheet or None if skipped.
    """
    # If the Compare sheet has no statements (blank except header), skip
    if df_compare.empty or df_compare['Statement'].dropna().empty:
        print(f"No statements found in Compare sheet '{sheet_name}'. Skipping...")
        return None

    # Get or create embeddings for the Mapped_PDF sheet
    main_embeddings = get_or_create_embeddings(df_main, model, sheet_name)

    # Encode compare statements
    print(f"Encoding compare statements for sheet '{sheet_name}'...")
    compare_statements = df_compare['Statement'].tolist()
    compare_embeddings = model.encode(compare_statements, convert_to_tensor=True, show_progress_bar=True)

    # Compute similarity (creates a matrix of shape [len(compare), len(main)])
    print(f"Computing similarity for sheet '{sheet_name}'...")
    similarity_matrix = util.pytorch_cos_sim(compare_embeddings, main_embeddings)
    best_scores, best_idxs = similarity_matrix.max(dim=1)

    # Set a threshold to discard matches below a certain score
    threshold = 0.1
    results = []

    # Build results
    for i in range(len(compare_statements)):
        score = best_scores[i].item()
        idx = best_idxs[i].item()
        if score >= threshold:
            row_main = df_main.iloc[idx]
            results.append({
                'Number': df_compare.iloc[i]['Number'],
                'Statement': compare_statements[i],
                'Matched Statement': row_main['Statement'],
                'Matched Document Reference': row_main['Document'],
                'Similarity Score': score,
                'Related PDF': row_main['PDF_name'],
                'Folder location': row_main['Folder location']
            })

    # Convert to DataFrame
    sheet_result_df = pd.DataFrame(results)

    # Write this sheet's results to the Excel file
    sheet_result_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Now apply the conditional formatting in the Excel
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # We assume the 'Similarity Score' is in column E (5th column)
    num_rows = len(sheet_result_df)
    if num_rows > 0:
        cell_range = f'E2:E{num_rows+1}'
        
        red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        worksheet.conditional_format(cell_range, {
            'type': 'cell',
            'criteria': '<',
            'value': 0.8,
            'format': red_format
        })
        
        orange_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
        worksheet.conditional_format(cell_range, {
            'type': 'cell',
            'criteria': 'between',
            'minimum': 0.8,
            'maximum': 0.95,
            'format': orange_format
        })
        
        green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        worksheet.conditional_format(cell_range, {
            'type': 'cell',
            'criteria': '>=',
            'value': 0.95,
            'format': green_format
        })

        # Set column widths
        worksheet.set_column('A:A', 8)
        worksheet.set_column('B:B', 50)
        worksheet.set_column('C:C', 50)
        worksheet.set_column('D:D', 65)
        worksheet.set_column('F:F', 13)
        worksheet.set_column('G:G', 22)
        percentage_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column('E:E', 14, percentage_format)

    return sheet_result_df

# ---------------------------------------------------------------------
# Main: read both sheets from Mapped_PDF and Compare, process them, update PDFs
# ---------------------------------------------------------------------
def main():
    start_time = time.time()

    # Names of the sheets we want to process
    sheets = ["ODB", "OMI", "IBM_DB", "IES", "SAP_DB", "SES"]

    # Load the model
    print("Loading SentenceTransformer model...")
    with tqdm(total=1, desc="Loading Model") as pbar:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        pbar.update(1)
    print("Model loaded!\n")

    # Prepare to write all results to a single Excel file with multiple sheets
    output_excel = 'Excel_file/Result.xlsx'
    # We use engine='xlsxwriter' for easier conditional formatting
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        
        # Process each sheet
        all_results = {}
        for sheet_name in sheets:
            print(f"=== Processing sheet: {sheet_name} ===")
            
            # Try reading each sheet from Mapped_PDF and Compare
            try:
                df_main_sheet = pd.read_excel('Excel_file/Mapped_PDF.xlsx', sheet_name=sheet_name)
            except:
                print(f"Sheet '{sheet_name}' not found in Mapped_PDF.xlsx. Skipping...")
                continue

            try:
                df_compare_sheet = pd.read_excel('Excel_file/Compare.xlsx', sheet_name=sheet_name, header=1, dtype={'Number': str}) # skip first row for Compare.xlsx
            except:
                print(f"Sheet '{sheet_name}' not found in Compare.xlsx. Skipping...")
                continue

            # If Mapped_PDF sheet is empty or missing 'Statement' column, skip
            if df_main_sheet.empty or 'Statement' not in df_main_sheet.columns:
                print(f"Sheet '{sheet_name}' in Mapped_PDF.xlsx is empty or invalid. Skipping...")
                continue
            
            # Process the sheet and write results
            sheet_result_df = process_sheet(sheet_name, df_main_sheet, df_compare_sheet, model, writer)
            if sheet_result_df is not None and not sheet_result_df.empty:
                all_results[sheet_name] = sheet_result_df
            print(df_compare_sheet) # check each dataframe
        # Once we're done, the writer will save the Excel file
        print(f"\nAll sheets processed. Results saved to {output_excel}.\n")

    # ===========================
    # PDF Annotation Update Phase
    # ===========================
    print("Clearing 'output/' folder...")
    output_dir = 'output'
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(output_dir)

    # For each sheet's results, update PDF annotations
    # Assume we have a dictionary of DataFrames, one for each sheet, named all_results
    # e.g. all_results = { "ODB": df_odb_results, "OMI": df_OMI_results }

    for sheet_name, result_df in all_results.items():
        print(f"\n=== Updating PDFs for sheet: {sheet_name} ===")

        for idx, row in result_df.iterrows():
            # Skip rows where the similarity score is below 0.8
            if row['Similarity Score'] < 0.8:
                print(f"Skipping row {idx} in sheet '{sheet_name}' (score < 0.8)")
                continue

            # Get the PDF name from the 'Related PDF' column
            pdf_name = row['Related PDF']
            if not isinstance(pdf_name, str) or not pdf_name.strip():
                print(f"Row {idx}: Invalid or empty PDF name. Skipping.")
                continue

            # Ensure the file has a .pdf extension
            if not pdf_name.lower().endswith('.pdf'):
                pdf_name += '.pdf'

            # For ODB sheet -> ODB_document/<pdf_name>, OMI sheet -> OMI_document/<pdf_name>
            pdf_path = os.path.join(f"Documents/{sheet_name}_document", pdf_name)

            # Check if the file exists
            if not os.path.exists(pdf_path):
                print(f"Row {idx}: PDF not found: {pdf_path}. Skipping.")
                continue

            # Use the compare.xlsx number for the content of each annotation
            annotation_content = str(row['Number'])

            print(f"Opening PDF: {pdf_path} ...")
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"Failed to open {pdf_path}: {e}")
                continue

            # Update all FreeText annotations
            for page in doc:
                annots = page.annots()
                if annots:
                    for annot in annots:
                        if "FreeText" in annot.type:
                            annot.set_info(content=annotation_content, title="Oracle")
                            annot.update()

            # Save the updated PDF into the output folder
            # Name it using {sheet}_{row['Number']}.pdf (adjust as needed)
            out_pdf_name = f"{sheet_name}_{row['Number']}.pdf"
            out_path = os.path.join("output", out_pdf_name)
            doc.save(out_path)
            doc.close()
            print(f"Saved updated PDF to: {out_path}")
    
    # Print total time
    end_time = time.time()
    print(f"\nAll done! Total time: {end_time - start_time:.3f} seconds.")


# ---------------------------------------------------------------------
# Run main if executed directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
