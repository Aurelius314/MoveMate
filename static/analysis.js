document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ Running analysis page script...");

    // 获取视频 URL
    fetch("/get_video")
    .then(response => response.json())
    .then(data => {
        console.log("✅ Server response from /get_video:", data);  // 🔍 检查 Flask 返回的 JSON

        if (data.video_url) {
            console.log("✅ Setting video source to:", data.video_url);
            const videoPlayer = document.getElementById("videoPlayer");
            videoPlayer.src = data.video_url;
            videoPlayer.load();
        } else {
            console.error("❌ No video URL received from server!");
        }
    })
    .catch(error => {
        console.error("❌ Error fetching video URL:", error);
    });

    function checkStatus() {
        fetch("/check_status")
        .then(response => response.json())
        .then(data => {
            if (!data.done) {
                setTimeout(checkStatus, 2000);  // 分析尚未完成，2 秒后再查询
                return;
            }
            if (data.error) {
                document.getElementById("loading-text").textContent = "Analysis failed: " + data.error;
                document.querySelector(".loader").style.display = "none";
                return;
            }
            if (data.image_url_1) document.getElementById("result-image-1").src = data.image_url_1;
            if (data.image_url_2) document.getElementById("result-image-2").src = data.image_url_2;
            document.getElementById("loading-container").style.display = "none";
            document.querySelector(".loader").style.display = "none";
            document.getElementById('result-container1').style.display = 'flex';
            document.getElementById('result-container2').style.display = 'flex';
        })
        .catch(error => {
            console.error("❌ Error checking status:", error);
            setTimeout(checkStatus, 5000);
        });
    }
    checkStatus();
});
