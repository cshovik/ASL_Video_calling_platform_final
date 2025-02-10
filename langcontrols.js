// controls.js

// Global variables for chat voice recognition.
var recognition;
var isRecognizing = false;

if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = true;       // Keep listening continuously.
  recognition.interimResults = false;  // Use final results only.
  recognition.lang = 'en-US';          // Change language as needed.
  recognition.onresult = function (event) {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        transcript += event.results[i][0].transcript;
      }
    }
    let chatInput = document.getElementById("chatInput");
    if (chatInput) {
      chatInput.value += transcript;
    }
  };
  recognition.onerror = function (event) {
    console.error("Speech recognition error:", event.error);
    isRecognizing = false;
    let voiceBtn = document.getElementById("voiceChatButton");
    if (voiceBtn) {
      voiceBtn.textContent = "🎤";
    }
  };
}

function initializeControls() {
  const controlsDiv = document.getElementById("controls");
  // Include a language dropdown for translation/TTS alongside the Speak button.
  controlsDiv.innerHTML = `
    <button id="muteButton" class="btn-mute">Mute</button>
    <button id="cameraButton" class="btn-camera">Camera Off</button>
    <button id="signDetectButton" class="btn-sign">Sign Detect On</button>
    <button id="leaveCallButton" class="btn-leave">Leave Call</button>
    <button id="chatButton" class="btn-chat">Chat</button>
    <!-- Text box for message input or ASL letters -->
    <input type="text" id="messageBox" placeholder="Message / ASL Letters" style="width:200px; margin-left:10px;">
    <!-- Language dropdown for translation/TTS -->
    <select id="translationLanguage" style="margin-left:10px;">
      <option value="hi">Hindi</option>
      <option value="en">English</option>
      <option value="fr">French</option>
      <option value="es">Spanish</option>
      <option value="de">German</option>
    </select>
    <button id="speechButton" class="btn-speech">Speak</button>
    <button id="clearButton" class="btn-clear">Clear</button>
  `;

  document.getElementById("muteButton").addEventListener("click", toggleMute);
  document.getElementById("cameraButton").addEventListener("click", toggleCamera);
  document.getElementById("signDetectButton").addEventListener("click", toggleSignDetection);
  document.getElementById("leaveCallButton").addEventListener("click", leaveCall);
  document.getElementById("speechButton").addEventListener("click", sendSpeech);
  document.getElementById("clearButton").addEventListener("click", clearMessage);
  document.getElementById("chatButton").addEventListener("click", toggleChatBox);

  // Create the chat box if it doesn't exist.
  if (!document.getElementById("chatBox")) {
    createChatBox();
  }
}

function toggleMute() {
  const audioTracks = localStream.getAudioTracks();
  if (audioTracks.length > 0) {
    audioTracks[0].enabled = !audioTracks[0].enabled;
    document.getElementById("muteButton").textContent = audioTracks[0].enabled ? "Mute" : "Unmute";
  }
}

function toggleCamera() {
  const videoTracks = localStream.getVideoTracks();
  if (videoTracks.length > 0) {
    videoTracks[0].enabled = !videoTracks[0].enabled;
    document.getElementById("cameraButton").textContent = videoTracks[0].enabled ? "Camera Off" : "Camera On";
  }
}

function toggleSignDetection() {
  detectASL = !detectASL;
  document.getElementById("signDetectButton").textContent = detectASL ? "Sign Detect Off" : "Sign Detect On";
  if (detectASL) {
    processASL();
  } else {
    const localWrapper = document.querySelector('.video-wrapper[data-id="local"]');
    if (localWrapper) {
      const canvas = localWrapper.querySelector("#aslCanvas");
      if (canvas) {
        canvas.remove();
      }
    }
  }
}

function leaveCall() {
  for (let id in peerConnections) {
    peerConnections[id].close();
  }
  window.location.reload();
}

/* --- Modified Speak Functionality (Translation/TTS) --- */
function sendSpeech() {
  const message = document.getElementById("messageBox").value;
  const language = document.getElementById("translationLanguage").value;
  if (message.trim().length > 0) {
    // Send text to the translation endpoint
    fetch("http://localhost:5000/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message, language: language })
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.error) {
          alert("Error: " + data.error);
          return;
        }
        // Update the text box with the translated text.
        document.getElementById("messageBox").value = data.translated_text;
        // Emit the chat message along with the audio URL.
        socket.emit("chat", {
          message: data.translated_text,
          name: window.userName,
          isSpeech: true,
          audio_url: data.audio_url
        });
        // Play the generated audio locally.
        let audio = new Audio(data.audio_url);
        audio.play().catch(err => console.error("Audio play error:", err));
      })
      .catch(error => {
        console.error("Error in translation:", error);
        alert("Something went wrong: " + error.message);
      });
  }
}

function clearMessage() {
  document.getElementById("messageBox").value = "";
}

function speakMessage(message) {
  const utterance = new SpeechSynthesisUtterance(message);
  window.speechSynthesis.speak(utterance);
}
window.speakMessage = speakMessage;

/* ----- Chat Box Functions ----- */
function toggleChatBox() {
  const chatBox = document.getElementById("chatBox");
  chatBox.style.display = (chatBox.style.display === "none" || chatBox.style.display === "") ? "flex" : "none";
}

function createChatBox() {
  const chatBox = document.createElement("div");
  chatBox.id = "chatBox";
  chatBox.className = "chat-box";
  chatBox.style.position = "fixed";
  chatBox.style.right = "20px";
  chatBox.style.bottom = "100px";
  chatBox.style.width = "300px";
  chatBox.style.height = "400px";
  chatBox.style.background = "#1e1e1e";
  chatBox.style.border = "1px solid #333";
  chatBox.style.borderRadius = "5px";
  chatBox.style.display = "none";
  chatBox.style.flexDirection = "column";
  chatBox.style.overflow = "hidden";
  chatBox.style.zIndex = "1000";
  chatBox.innerHTML = `
    <div class="chat-header" style="background: #333; padding: 10px; color: #fff; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444;">
      <span>Chat</span>
      <button id="closeChat" style="background: transparent; border: none; color: #fff; cursor: pointer;">X</button>
    </div>
    <div class="chat-messages" style="flex: 1; padding: 10px; overflow-y: auto; color: #fff; font-size: 14px;"></div>
    <div class="chat-input" style="display: flex; border-top: 1px solid #444;">
      <input type="text" id="chatInput" placeholder="Type your message here..." style="flex: 1; padding: 10px; border: none; outline: none; background: #222; color: #fff;">
      <button id="voiceChatButton" style="padding: 10px; background: #555; border: none; color: #fff; cursor: pointer;">🎤</button>
      <button id="sendChat" style="padding: 10px; background: #2196F3; border: none; color: #fff; cursor: pointer;">Send</button>
    </div>
  `;
  document.body.appendChild(chatBox);
  document.getElementById("closeChat").addEventListener("click", function () {
    chatBox.style.display = "none";
  });
  document.getElementById("sendChat").addEventListener("click", sendChatMessage);
  document.getElementById("chatInput").addEventListener("keyup", function (e) {
    if (e.key === "Enter") {
      sendChatMessage();
    }
  });
  document.getElementById("voiceChatButton").addEventListener("click", toggleChatVoice);
}

function toggleChatVoice() {
  if (!recognition) {
    alert("Speech recognition is not supported in this browser.");
    return;
  }
  const voiceBtn = document.getElementById("voiceChatButton");
  if (!isRecognizing) {
    recognition.start();
    isRecognizing = true;
    voiceBtn.textContent = "🛑";
  } else {
    recognition.stop();
    isRecognizing = false;
    voiceBtn.textContent = "🎤";
  }
}

function sendChatMessage() {
  const chatInput = document.getElementById("chatInput");
  const message = chatInput.value.trim();
  if (message.length === 0) return;
  const data = { message: message, name: window.userName };
  if (window.socket) {
    window.socket.emit("chat", data);
  }
  appendChatMessage(data);
  chatInput.value = "";
}

function appendChatMessage(data) {
  const chatBox = document.getElementById("chatBox");
  if (!chatBox) return;
  const messagesContainer = chatBox.querySelector(".chat-messages");
  const messageElem = document.createElement("div");
  messageElem.className = "chat-message";
  messageElem.textContent = data.name + ": " + data.message;
  messagesContainer.appendChild(messageElem);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
