let selectedFile = null;  // Global variable to store the selected file

// Function to handle the actual file upload
function uploadFile(file) {
    let formData = new FormData();
    formData.append('file', file);

    fetch('/success', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        // Replace the page content with the server's response (Acknowledgement page)
        document.body.innerHTML = data;
    })
    .catch(error => console.error('Error:', error));
}

function dragEnterHandler(ev) {
    ev.preventDefault();
    console.log("File(s) in drop zone");
}

function dragOverHandler(ev) {
    ev.preventDefault();
}

function dragLeaveHandler(ev) {
    ev.preventDefault();
    console.log("File(s) left the drop zone");
}

function dropHandler(ev) {
    ev.preventDefault();  // Prevent default behavior (e.g., file opening in browser)
    console.log("File(s) dropped");

    let file;
    if (ev.dataTransfer.items) {
        // Modern browsers: get the first file
        for (let i = 0; i < ev.dataTransfer.items.length; i++) {
            if (ev.dataTransfer.items[i].kind === "file") {
                file = ev.dataTransfer.items[i].getAsFile();
                break;
            }
        }
    } else {
        file = ev.dataTransfer.files[0];
    }

    if (file) {
        // Check that the file is an Excel file (ends with .xlsx)
        if (!file.name.toLowerCase().endsWith('.xlsx')) {
            swal("Invalid File Type", "Please upload an Excel file (.xlsx)", "error");
            return;
        }
        selectedFile = file;
        document.getElementById('filePreview').innerText = "Selected file: " + file.name;
        document.getElementById('submitBtn').disabled = false;
    }
}

// Clicking the drop zone opens the file dialog
document.getElementById("drop_zone").addEventListener("click", function() {
    document.getElementById("fileInput").click();
});

// When a file is selected through the dialog, check if it is an Excel file
document.getElementById("fileInput").addEventListener("change", function(event) {
    let file = event.target.files[0];
    if (file) {
        if (!file.name.toLowerCase().endsWith('.xlsx')) {
            swal("Invalid File Type", "Please upload an Excel file (.xlsx)", "error");
            return;
        }
        selectedFile = file;
        document.getElementById('filePreview').innerText = "Selected file: " + file.name;
        document.getElementById('submitBtn').disabled = false;
    }
});

// Submit button click event: upload the stored file
document.getElementById("submitBtn").addEventListener("click", function() {
    if (selectedFile) {
        uploadFile(selectedFile);
    } else {
        swal("No File Selected", "Please select a file first", "warning");
    }
});

// Submit button click event: upload the stored file
document.getElementById("submitBtn").addEventListener("click", function() {
  if (selectedFile) {
      // Show the spinner immediately
      document.getElementById("spinner").style.display = "block";
      // Optionally, disable the submit button to prevent multiple clicks
      document.getElementById("submitBtn").disabled = true;

      // Proceed to upload the file
      uploadFile(selectedFile);
  } else {
      swal("No File Selected", "Please select a file first", "warning");
  }
});



