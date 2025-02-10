// videoCall.js

const socket = io();
window.socket = socket; // Expose socket globally

let localStream;
let peerConnections = {};
let userName = "";
let detectASL = false;

function startVideoCall(name) {
  userName = name;
  window.userName = name;
  document.getElementById("app").innerHTML = `
    <div id="videoContainer" class="video-grid"></div>
    <div id="controls" class="control-bar"></div>
  `;
  
  initializeVideoGrid();
  initializeControls();
  
  navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then((stream) => {
      localStream = stream;
      addVideoStream(userName, localStream, true, "local");
      socket.emit("new-user", { name: userName });
    })
    .catch((err) => console.error("Error accessing media devices:", err));
  
  socket.on("users", (users) => {
    users.forEach((user) => {
      if (user.id !== socket.id && !peerConnections[user.id]) {
        // Initiate call only if our socket id is lexicographically lower.
        if (socket.id < user.id) {
          initiateCall(user.id, user.name);
        }
      }
    });
  });
  
  socket.on("offer", async (data) => {
    if (peerConnections[data.from]) return;
    const pc = createPeerConnection(data.from, data.name);
    try {
      await pc.setRemoteDescription(new RTCSessionDescription(data.offer));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit("answer", { target: data.from, answer, name: userName });
    } catch (e) {
      console.error("Error handling offer:", e);
    }
  });
  
  socket.on("answer", (data) => {
    const pc = peerConnections[data.from];
    if (pc && pc.signalingState === "have-local-offer") {
      pc.setRemoteDescription(new RTCSessionDescription(data.answer))
        .catch((err) => console.error("Error setting remote description:", err));
    }
  });
  
  socket.on("candidate", (data) => {
    const pc = peerConnections[data.from];
    if (pc) {
      pc.addIceCandidate(new RTCIceCandidate(data.candidate))
        .catch((err) => console.error("Error adding ICE candidate:", err));
    }
  });
  
  socket.on("user-disconnected", (id) => {
    if (peerConnections[id]) {
      peerConnections[id].close();
      delete peerConnections[id];
      removeVideoStream(id);
    }
  });
  
  socket.on("chat", (data) => {
    appendChatMessage(data);
    // If a chat message is flagged as speech and includes an audio URL, play it.
    if (data.isSpeech && data.audio_url) {
      let audio = new Audio(data.audio_url);
      audio.play().catch(err => console.error("Audio play error:", err));
    } else if (data.isSpeech) {
      speakMessage(data.message);
    }
  });
}

function initiateCall(targetId, targetName) {
  const pc = createPeerConnection(targetId, targetName);
  pc.createOffer()
    .then((offer) => pc.setLocalDescription(offer))
    .then(() => {
      socket.emit("offer", { target: targetId, offer: pc.localDescription, name: userName });
    })
    .catch((err) => console.error("Error initiating call:", err));
}

function createPeerConnection(targetId, targetName) {
  const pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
  peerConnections[targetId] = pc;
  
  localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));
  
  pc.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit("candidate", { target: targetId, candidate: event.candidate });
    }
  };
  
  pc.ontrack = (event) => {
    let remoteStream;
    if (event.streams && event.streams[0]) {
      remoteStream = event.streams[0];
    } else {
      remoteStream = new MediaStream();
      remoteStream.addTrack(event.track);
    }
    addVideoStream(targetName, remoteStream, false, targetId);
  };
  
  return pc;
}

function updateLocalVideo(name, stream) {
  const localVideoElem = document.querySelector(`.video-wrapper[data-id="local"] video`);
  if (localVideoElem) {
    localVideoElem.srcObject = stream;
  }
}

function removeVideoStream(id) {
  const videoElem = document.querySelector(`.video-wrapper[data-id="${id}"]`);
  if (videoElem) {
    videoElem.remove();
    updateVideoLayout();
  }
}

function initializeVideoGrid() {
  updateVideoLayout();
}

function addVideoStream(name, stream, isLocal = false, id = "local") {
  let existing = document.querySelector(`.video-wrapper[data-id="${id}"]`);
  if (existing) {
    const video = existing.querySelector("video");
    if (video.srcObject !== stream) {
      video.srcObject = stream;
      video.play().catch((err) => console.error("Error playing video:", err));
    }
    const label = existing.querySelector(".video-label");
    label.textContent = name;
    return;
  }
  
  const videoWrapper = document.createElement("div");
  videoWrapper.className = "video-wrapper";
  videoWrapper.dataset.id = id;
  
  const videoElem = document.createElement("video");
  videoElem.autoplay = true;
  videoElem.playsInline = true;
  videoElem.srcObject = stream;
  if (isLocal) videoElem.muted = true;
  videoWrapper.appendChild(videoElem);
  
  const nameLabel = document.createElement("div");
  nameLabel.className = "video-label";
  nameLabel.textContent = name;
  videoWrapper.appendChild(nameLabel);
  
  document.getElementById("videoContainer").appendChild(videoWrapper);
  updateVideoLayout();
}

function updateVideoLayout() {
  const container = document.getElementById("videoContainer");
  const count = container.children.length;
  if (count === 1) {
    container.style.gridTemplateColumns = "1fr";
  } else if (count === 2) {
    container.style.gridTemplateColumns = "1fr 1fr";
  } else if (count > 2) {
    container.style.gridTemplateColumns = "repeat(auto-fit, minmax(300px, 1fr))";
  }
}

/* --- Sign Detection (ASL) --- */
function processASL() {
  if (!detectASL) return;
  const localWrapper = document.querySelector(`.video-wrapper[data-id="local"]`);
  if (!localWrapper) return;
  const videoElem = localWrapper.querySelector("video");
  if (!videoElem || videoElem.readyState !== videoElem.HAVE_ENOUGH_DATA) {
    return setTimeout(processASL, 300);
  }
  let aslCanvas = localWrapper.querySelector("#aslCanvas");
  if (!aslCanvas) {
    aslCanvas = document.createElement("canvas");
    aslCanvas.id = "aslCanvas";
    aslCanvas.style.position = "absolute";
    aslCanvas.style.top = "0";
    aslCanvas.style.left = "0";
    aslCanvas.style.width = "100%";
    aslCanvas.style.height = "100%";
    aslCanvas.style.pointerEvents = "none";
    localWrapper.appendChild(aslCanvas);
  }
  const context = aslCanvas.getContext("2d");
  aslCanvas.width = videoElem.videoWidth;
  aslCanvas.height = videoElem.videoHeight;
  context.drawImage(videoElem, 0, 0, aslCanvas.width, aslCanvas.height);
  aslCanvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");
    fetch("/detect", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((detections) => {
        context.clearRect(0, 0, aslCanvas.width, aslCanvas.height);
        context.drawImage(videoElem, 0, 0, aslCanvas.width, aslCanvas.height);
        detections.forEach((det) => {
          context.strokeStyle = "red";
          context.lineWidth = 2;
          context.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
          context.fillStyle = "red";
          context.font = "16px Arial";
          context.fillText(`${det.label} (${det.confidence.toFixed(2)})`, det.x1, det.y1 - 5);
        });
        // Append the first detected letter to the message box if not already there.
        if (detections.length > 0) {
          let letter = detections[0].label;
          const messageBox = document.getElementById("messageBox");
          if (messageBox) {
            let currentText = messageBox.value;
            if (currentText.slice(-1) !== letter) {
              messageBox.value += letter;
            }
          }
        }
      })
      .catch((err) => console.error("Error in ASL detection:", err));
  });
  setTimeout(processASL, 300);
}
window.processASL = processASL;
