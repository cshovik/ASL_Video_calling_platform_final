const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const multer = require("multer");
const FormData = require("form-data");
const axios = require("axios");

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const PORT = 3000;

// Store connected users as objects with id and name
const connectedUsers = {};

// Multer setup for handling file uploads
const upload = multer();

app.use(express.static(__dirname));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Proxy route for ASL detection (Flask API)
app.post("/detect", upload.single("frame"), async (req, res) => {
  try {
    const formData = new FormData();
    formData.append("frame", req.file.buffer, "frame.jpg");

    const flaskResponse = await axios.post("http://localhost:5000/detect", formData, {
      headers: formData.getHeaders(),
      timeout: 5000 // Prevents hanging if Flask server is down
    });

    res.status(flaskResponse.status).send(flaskResponse.data);
  } catch (error) {
    console.error("Error proxying request:", error.message);
    res.status(500).send({ error: "Detection failed", details: error.message });
  }
});

// WebRTC signaling
io.on("connection", (socket) => {
  console.log(`User connected: ${socket.id}`);

  // Listen for new-user event to store user info
  socket.on("new-user", (data) => {
    connectedUsers[socket.id] = { id: socket.id, name: data.name };
    io.emit("users", Object.values(connectedUsers));
  });

  // Relay offer along with sender’s name
  socket.on("offer", (data) => {
    const sender = connectedUsers[socket.id] || { id: socket.id, name: "Anonymous" };
    io.to(data.target).emit("offer", { from: socket.id, offer: data.offer, name: sender.name });
  });

  // Relay answer back to caller
  socket.on("answer", (data) => {
    io.to(data.target).emit("answer", { from: socket.id, answer: data.answer });
  });

  // Relay ICE candidate
  socket.on("candidate", (data) => {
    io.to(data.target).emit("candidate", { from: socket.id, candidate: data.candidate });
  });

  // New: Relay chat messages.
  socket.on("chat", (data) => {
    socket.broadcast.emit("chat", data);
  });

  // On disconnect, remove user and notify others
  socket.on("disconnect", () => {
    console.log(`User disconnected: ${socket.id}`);
    delete connectedUsers[socket.id];
    io.emit("users", Object.values(connectedUsers));
    io.emit("user-disconnected", socket.id);
  });
});

server.listen(PORT, () => {
  console.log(`Node.js server running on http://localhost:${PORT}`);
});
