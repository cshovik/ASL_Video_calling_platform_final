// controls.js

// Global variables for chat voice recognition.
var recognition;
var isRecognizing = false;

// Initialize SpeechRecognition if available.
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
    // Append the recognized transcript to the chat input.
    let chatInput = document.getElementById("chatInput");
    if (chatInput) {
      // You can choose to append a space then the new transcript.
      chatInput.value += transcript;
    }
  };
  recognition.onerror = function (event) {
    console.error("Speech recognition error: ", event.error);
    // Reset the toggle on error.
    isRecognizing = false;
    let voiceBtn = document.getElementById("voiceChatButton");
    if (voiceBtn) {
      voiceBtn.textContent = "🎤";
    }
  };
}

function initializeControls() {
  const controlsDiv = document.getElementById("controls");
  controlsDiv.innerHTML = `
    <button id="muteButton" class="btn-mute">Mute</button>
    <button id="cameraButton" class="btn-camera">Camera Off</button>
    <button id="signDetectButton" class="btn-sign">Sign Detect On</button>
    <button id="leaveCallButton" class="btn-leave">Leave Call</button>
    <button id="chatButton" class="btn-chat">Chat</button>
    <!-- Front text box for gesture input & speech conversion -->
    <input type="text" id="messageBox" placeholder="Message / ASL Letters" style="width:200px; margin-left:10px;">
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

  // Create the chat box if not already present.
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
    // Remove the canvas overlay if sign detection is turned off.
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
  // Close all peer connections and reload the page.
  for (let id in peerConnections) {
    peerConnections[id].close();
  }
  window.location.reload();
}

// Front text box's Speak button function:
// Convert text (from "messageBox") to speech locally and broadcast it with an isSpeech flag.
function sendSpeech() {
  const message = document.getElementById("messageBox").value;
  if (message.trim().length > 0) {
    // Convert text to speech locally.
    speakMessage(message);
    // Broadcast the message so that the other device receives it and it appears in the chat history.
    socket.emit("chat", { message: message, name: window.userName, isSpeech: true });
  }
}

// Clear the front text box.
function clearMessage() {
  document.getElementById("messageBox").value = "";
}

// Use the Web Speech API to speak a message.
function speakMessage(message) {
  const utterance = new SpeechSynthesisUtterance(message);
  window.speechSynthesis.speak(utterance);
}
window.speakMessage = speakMessage;

/* ===== Chat Box Functionality ===== */

// Toggle (show/hide) the chat box.
function toggleChatBox() {
  const chatBox = document.getElementById("chatBox");
  // Use "flex" to preserve the flex layout.
  chatBox.style.display = (chatBox.style.display === "none" || chatBox.style.display === "") ? "flex" : "none";
}

// Create and insert the chat box into the DOM.
function createChatBox() {
  const chatBox = document.createElement("div");
  chatBox.id = "chatBox";
  chatBox.className = "chat-box";
  
  // Fixed-size rectangular card with flex layout.
  chatBox.style.position = "fixed";
  chatBox.style.right = "20px";
  chatBox.style.bottom = "100px";
  chatBox.style.width = "300px";
  chatBox.style.height = "400px"; // Fixed height
  chatBox.style.background = "#1e1e1e";
  chatBox.style.border = "1px solid #333";
  chatBox.style.borderRadius = "5px";
  chatBox.style.display = "none"; // Hidden by default.
  chatBox.style.flexDirection = "column";
  chatBox.style.overflow = "hidden"; // Ensure inner content does not break the fixed size.
  chatBox.style.zIndex = "1000";
  
  // Inner HTML structure:
  // - A header with a close button.
  // - A messages container that is scrollable.
  // - A chat input area with an input field, a voice icon button, and a send button.
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

  // Close chat box when X is clicked.
  document.getElementById("closeChat").addEventListener("click", function () {
    chatBox.style.display = "none";
  });

  // Add event listeners for chat send and voice button.
  document.getElementById("sendChat").addEventListener("click", sendChatMessage);
  document.getElementById("chatInput").addEventListener("keyup", function (e) {
    if (e.key === "Enter") {
      sendChatMessage();
    }
  });
  document.getElementById("voiceChatButton").addEventListener("click", toggleChatVoice);
}

// Toggle voice recognition for the chat input.
function toggleChatVoice() {
  if (!recognition) {
    alert("Speech recognition is not supported in this browser.");
    return;
  }
  const voiceBtn = document.getElementById("voiceChatButton");
  if (!isRecognizing) {
    recognition.start();
    isRecognizing = true;
    // Change the icon to indicate that voice recognition is active.
    voiceBtn.textContent = "🛑"; // Press again to stop.
  } else {
    recognition.stop();
    isRecognizing = false;
    voiceBtn.textContent = "🎤";
  }
}

// Send a chat message from the chat box.
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

// Append a chat message to the chat messages container.
function appendChatMessage(data) {
  const chatBox = document.getElementById("chatBox");
  if (!chatBox) return;
  const messagesContainer = chatBox.querySelector(".chat-messages");
  const messageElem = document.createElement("div");
  messageElem.className = "chat-message";
  messageElem.textContent = data.name + ": " + data.message;
  messagesContainer.appendChild(messageElem);
  // Auto-scroll to the bottom so the latest message is visible.
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

