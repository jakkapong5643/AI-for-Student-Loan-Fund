const fileUpload = document.getElementById('file-upload');
const status = document.getElementById('status');
const resetBtn = document.getElementById('reset-btn');
const uploadBox = document.querySelector('.upload-box');
let uploadedFile = null;

const allowedTypes = [
  'application/pdf',
  'image/jpeg',
  'image/png'
];

function validateFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowedTypes.includes(file.type)) return false;
  if (!['pdf', 'jpg', 'jpeg', 'png'].includes(ext)) return false;
  return true;
}

function fakeApiCheck(file) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const isValid = Math.random() > 0.4;
      resolve(isValid);
    }, 2500);
  });
}

function updateStatus(text, statusClass) {
  status.textContent = text;
  status.className = 'status ' + statusClass;
}

function reset() {
  fileUpload.value = '';
  uploadedFile = null;
  updateStatus('Please upload your document to start the Check.', 'checking');
  resetBtn.style.display = 'none';
}

uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');

  if (e.dataTransfer.files.length > 0) {
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }
});

async function handleFile(file) {
  if (!validateFile(file)) {
    updateStatus('ชนิดไฟล์ไม่ถูกต้อง! รองรับเฉพาะ PDF, JPG, PNG เท่านั้น', 'error');
    resetBtn.style.display = 'inline-block';
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    updateStatus('ไฟล์ใหญ่เกินไป (เกิน 10MB)', 'error');
    resetBtn.style.display = 'inline-block';
    return;
  }

  uploadedFile = file;
  updateStatus('กำลังตรวจสอบเอกสาร...', 'checking');
  resetBtn.style.display = 'none';

  const isValid = await fakeApiCheck(file);

  if (isValid) {
    updateStatus('เอกสารถูกต้อง ', 'success');
  } else {
    updateStatus('พบข้อผิดพลาดในเอกสาร กรุณาตรวจสอบและส่งใหม่', 'error');
  }
  resetBtn.style.display = 'inline-block';
}

fileUpload.addEventListener('change', () => {
  if (!fileUpload.files.length) return;
  handleFile(fileUpload.files[0]);
});

resetBtn.addEventListener('click', reset);

reset();
