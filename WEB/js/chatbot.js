const chatBox = document.getElementById('chat-box');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const clearBtn = document.getElementById('clear-btn');

function appendMessage(text, sender, loading = false) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message ' + sender;
  if (loading) msgDiv.classList.add('loading');
  msgDiv.textContent = text;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;
  appendMessage(text, 'user');
  userInput.value = '';
  appendMessage('กำลังคิดคำตอบ...', 'bot', true);

  setTimeout(() => {
    const loadingMsg = chatBox.querySelector('.message.bot.loading:last-child');
    if (loadingMsg) chatBox.removeChild(loadingMsg);

    const answers = [
      'คุณสามารถส่งเอกสารผ่านหน้า "ตรวจสอบเอกสาร" ได้เลยค่ะ',
      'ระยะเวลาการอนุมัติขึ้นอยู่กับความครบถ้วนของเอกสาร',
      'หากต้องการความช่วยเหลือเพิ่มเติม ติดต่อเจ้าหน้าที่ กยศ. ได้ที่ support@gysc.go.th',
      'กรุณาตรวจสอบเอกสารให้ครบถ้วนก่อนส่ง เพื่อเร่งกระบวนการอนุมัติ',
    ];
    const answer = answers[Math.floor(Math.random() * answers.length)];
    appendMessage(answer, 'bot');
  }, 1500);
});

clearBtn.addEventListener('click', () => {
  chatBox.innerHTML = '';
  userInput.focus();
});
