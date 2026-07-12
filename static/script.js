const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const video = document.getElementById("videoFeed");

startBtn.addEventListener("click", () => {

    video.src = "/video";

    video.style.display = "block";

    startBtn.style.display = "none";
    stopBtn.style.display = "inline-block";

});

stopBtn.addEventListener("click", () => {

    video.src = "";

    video.style.display = "none";

    stopBtn.style.display = "none";
    startBtn.style.display = "inline-block";

});