//const dropzone = document.getElementsByClassName("drop-zone");

function dragEnterHandler(ev) {
    ev.preventDefault();
    console.log("File(s) in drop zone");
  }

  // Called repeatedly by the browser as you move the mouse over the zone,
  // but we only need it to allow dropping, so no logging here.
function dragOverHandler(ev) {
    ev.preventDefault();
  }

  // Optional: if you want to know when the user drags out of the zone
function dragLeaveHandler(ev) {
    ev.preventDefault();
    console.log("File(s) left the drop zone");
  }

function dropHandler(ev) {
    console.log("File(s) dropped");
    ev.preventDefault(); // Prevent file from being opened by the browser

    if (ev.dataTransfer.items) {
      // For modern browsers
      [...ev.dataTransfer.items].forEach((item, i) => {
        if (item.kind === "file") {
          const file = item.getAsFile();
          console.log(`… file[${i}].name = ${file.name}`);
        }
      });
    } else {
      // For older browsers
      [...ev.dataTransfer.files].forEach((file, i) => {
        console.log(`… file[${i}].name = ${file.name}`);
      });
    }
  }