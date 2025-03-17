import os
import subprocess
from flask import Flask, render_template, request, send_from_directory, send_file
import io
import zipfile

app = Flask(__name__)

# Define the upload folder (absolute path recommended)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Excel_file')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        # Grab the uploaded file from the request
        uploaded_file = request.files['file']
        if not uploaded_file:
            return "No file was uploaded!", 400

        # Save the file as Compare.xlsx in Excel_file folder
        compare_path = os.path.join(UPLOAD_FOLDER, 'Compare.xlsx')
        uploaded_file.save(compare_path)

        # Optional: verify it actually saved
        if not os.path.exists(compare_path):
            return "Error: Compare.xlsx was not created!", 500

        # Path to your environment’s python.exe
        # For example: r"C:\Users\USER\Desktop\Autocomply\env\Scripts\python.exe"
        python_executable = os.path.join(os.getcwd(), "env", "Scripts", "python.exe") #use .venv if on work laptop
        script_path = os.path.join(os.getcwd(), "Complier.py")

        try:
            subprocess.run([python_executable, script_path],
                           capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            return f"Error occurred while processing file:\n{e.stderr}", 500

        return render_template("Acknowledgement.html", name='Compare.xlsx')

# download result file
@app.route('/download_result')
def download_result():
    upload_folder = os.path.join(os.getcwd(), 'Excel_file')
    # Option 1: Named arguments
    return send_from_directory(
        directory=upload_folder, 
        path="Result.xlsx", 
        as_attachment=True
    )

# download Zip file of output folder
@app.route('/download_output_zip')
def download_output_zip():
    output_folder = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_folder):
        return "Output folder does not exist!"
    
    # Create an in-memory zip file
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Walk through the output folder, add each file
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Use a relative path so the zip doesn't have absolute directories
                arcname = os.path.relpath(file_path, output_folder)
                zf.write(file_path, arcname)
    
    memory_file.seek(0) # Important: go back to the start of the BytesIO

    return send_file(
        memory_file,
        mimetype='application/zip',
        download_name='output.zip',
        as_attachment=True
    )



if __name__ == '__main__':
    app.run(debug=True)
