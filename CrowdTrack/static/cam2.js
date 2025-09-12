// 시간 업데이트
function updateTime() {
    const now = new Date();
    document.getElementById('current-time').textContent = now.toLocaleTimeString('ko-KR');
}

let lastEventTime = Date.now(); // 마지막 이벤트 발생 시간

// 실시간 데이터 업데이트
function updateStats() {
    fetch('/cam2/api/current_tracking_count')
    .then(response => response.json())
    .then(data => {
        const currentTime = Date.now();
        const timeElapsed = currentTime - lastEventTime;

        // 이벤트가 발생했을 때 tracking count 업데이트
        if (timeElapsed <= 2000) {
            document.getElementById('tracking-count').textContent = data.tracking_count + '명';
        }

        // 2초 이상 변화가 없으면 0으로 초기화
        if (timeElapsed > 2000) {
            document.getElementById('tracking-count').textContent = '0명';
        }

        // 마지막 이벤트 발생 시간 갱신
        lastEventTime = currentTime;
    })
    .catch(error => console.error('Error fetching current_tracking_count:', error));


    fetch('/cam2/api/events_total')
    .then(response => response.json())
    .then(data => {
        // 서버에서 전달한 이벤트 총합을 화면에 표시
        document.getElementById('total-events').textContent = data.events_total + '건';
    })
    .catch(error => console.error('Error fetching events_total:', error));
    
    fetch('/cam2/api/zone_detections')
        .then(response => response.json())
        .catch(error => console.error('Error fetching stats:', error));
        
    fetch('/cam2/api/alerts')
        .then(response => response.json())
        .then(data => {
            const alertsContainer = document.getElementById('alerts-container');
            alertsContainer.innerHTML = '';
            data.alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert-item';
                alertDiv.textContent = alert;
                alertsContainer.appendChild(alertDiv);
            });
        })
        .catch(error => console.error('Error fetching alerts:', error));
}

// 비디오 오류 처리
document.getElementById('main-video').onerror = function() {
    console.log('비디오 로드 오류 발생');
    setTimeout(() => {
        this.src = this.src; // 재시도
    }, 5000);
};

// 초기화
updateTime();
setInterval(updateTime, 1000);
setInterval(updateStats, 1000);
updateStats();
