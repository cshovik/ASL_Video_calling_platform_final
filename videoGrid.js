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
