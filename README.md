# **Automated TOR Compliance creation project**

This project is a flask-based web application that enable users to upload their TOR file (In .xlsx format) and compare it against a template file (pre-loaded) to generate a resulting compliance file. The output is color-coded to reflect the similarity score between input statement and reference statement.

## Tech stack

- Python
- Flask
- HTML
- CSS
- JavaScript

## Requirements

- Refer to the requirements.txt file
- Not every library is required from requirements.txt file due to some libary being used only in testing

## Features

- Drag and drop file upload with friendly UX/UI
- Semantic similarity check
- Color-coded output based on similarity score
- Downloadable result file

## Usage

git clone https://github.com/yourusername/autocomply.git

cd autocomply

python3 -m venv env

source env/bin/activate

pip install -r requirements.txt

python main.py

Navigate to "http://127.0.0.1:5000/" in your browser

Follow instruction on the browser to input/download your file

## License
This project is licensed under the MIT License. 