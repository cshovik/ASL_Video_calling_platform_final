// videoCall.js

const socket = io();
window.socket = socket; // Expose the socket globally for use in controls.js

let localStream;
let peerConnections = {};
let userName = "";
let detectASL = false;

function startVideoCall(name) {
  userName = name;
  window.userName = name; // Make the user name available globally
  document.getElementById("app").innerHTML = `
    <div id="videoContainer" class="video-grid"></div>
    <div id="controls" class="control-bar"></div>
  `;

  initializeVideoGrid();
  initializeControls();

  // Get local media and add your own video to the grid.
  navigator.mediaDevices
    .getUserMedia({ video: true, audio: true })
    .then((stream) => {
      localStream = stream;
      addVideoStream(userName, localStream, true, "local");
      // Tell the server about our new user.
      socket.emit("new-user", { name: userName });
    })
    .catch((err) => console.error("Error accessing media devices:", err));

  // When receiving an updated user list, initiate calls only to those with a greater socket ID.
  socket.on("users", (users) => {
    users.forEach((user) => {
      if (user.id !== socket.id && !peerConnections[user.id]) {
        // Only initiate the call if my socket id is lexicographically lower than theirs.
        if (socket.id < user.id) {
          initiateCall(user.id, user.name);
        }
      }
    });
  });

  // When receiving an offer from a remote peer.
  socket.on("offer", async (data) => {
    if (peerConnections[data.from]) {
      console.warn("Already connected with", data.from);
      return;
    }
    const pc = createPeerConnection(data.from, data.name);
    try {
      await pc.setRemoteDescription(new RTCSessionDescription(data.offer));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit("answer", { target: data.from, answer, name: userName });
    } catch (e) {
      console.error("Error handling offer from", data.from, ":", e);
    }
  });

  // When receiving an answer for an offer we sent.
  socket.on("answer", (data) => {
    const pc = peerConnections[data.from];
    if (pc) {
      if (pc.signalingState === "have-local-offer") {
        pc
          .setRemoteDescription(new RTCSessionDescription(data.answer))
          .catch((err) =>
            console.error("Error setting remote description in answer:", err)
          );
      } else {
        console.warn("Received answer but signaling state is:", pc.signalingState);
      }
    }
  });

  // When receiving ICE candidates.
  socket.on("candidate", (data) => {
    const pc = peerConnections[data.from];
    if (pc) {
      pc
        .addIceCandidate(new RTCIceCandidate(data.candidate))
        .catch((err) => console.error("Error adding received ICE candidate", err));
    }
  });

  // When a user disconnects, remove its video and close its connection.
  socket.on("user-disconnected", (id) => {
    if (peerConnections[id]) {
      peerConnections[id].close();
      delete peerConnections[id];
      removeVideoStream(id);
    }
  });

  // Updated Chat Message Listener:
  // Append incoming chat messages and, if the message was sent via the Speech button, also speak it.
  socket.on("chat", (data) => {
    appendChatMessage(data);
    if (data.isSpeech) {
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
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });
  peerConnections[targetId] = pc;

  // Add local stream tracks.
  localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

  pc.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit("candidate", { target: targetId, candidate: event.candidate });
    }
  };

  // ontrack handler: ensure we get a proper remote MediaStream.
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

// --- ASL detection processing ---
// This function captures frames from the local video element, sends them to the ASL detection server,
// overlays detection results on the local video, and appends detected letters to the front text box.
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
        // Append the first detected letter to the front text box (if not already present at the end)
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

// Expose processASL globally so that controls.js can access it.
window.processASL = processASL;


