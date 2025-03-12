import os
import subprocess # for creating new child process
from flask import Flask, render_template, request, send_from_directory

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

if __name__ == '__main__':
    app.run(debug=True)
