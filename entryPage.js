// entryPage.js
function showEntryPage() {
  document.getElementById("app").innerHTML = `
    <div class="entry-container">
      <div class="entry-box">
        <img src="image.png" alt="SC Logo" class="sc-logo"> <!-- Added SC Logo -->
        <div class="video-preview" id="videoPreviewContainer">
          <video id="videoPreview" autoplay playsinline muted></video>
        </div>
        <h2>SC Video Calling Platform</h2>
        <input type="text" id="userName" placeholder="Enter your name" required>
        <div class="controls">
          <button id="toggleCamera" class="btn-toggle">Camera Off</button>
          <button id="toggleMic" class="btn-toggle">Mute</button>
        </div>
        <button id="joinCallBtn" class="btn-join">Join Now</button>
      </div>
    </div>
  `;

  const videoElem = document.getElementById("videoPreview");
  const cameraButton = document.getElementById("toggleCamera");
  const micButton = document.getElementById("toggleMic");

  navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then((stream) => {
      videoElem.srcObject = stream;
      window.localStream = stream;
    })
    .catch(() => alert("Please allow access to camera and microphone."));

  cameraButton.addEventListener("click", () => {
    const videoTrack = window.localStream.getVideoTracks()[0];
    videoTrack.enabled = !videoTrack.enabled;
    cameraButton.textContent = videoTrack.enabled ? "Camera Off" : "Camera On";
  });

  micButton.addEventListener("click", () => {
    const audioTrack = window.localStream.getAudioTracks()[0];
    audioTrack.enabled = !audioTrack.enabled;
    micButton.textContent = audioTrack.enabled ? "Mute" : "Unmute";
  });

  document.getElementById("joinCallBtn").addEventListener("click", () => {
    const name = document.getElementById("userName").value.trim();
    if (!name) {
      alert("Please enter your name");
      return;
    }
    startVideoCall(name);
  });
}

showEntryPage();


