let currentCamera = 'main';

// 시간 업데이트
function updateTime() {
    const now = new Date();
    const timeString = now.toLocaleString('ko-KR', {
        year: 'numeric', month: 'numeric', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true
    });
    document.getElementById('current-time').textContent = timeString;
}

// overlay 위치 계산
function calculateZonePositions() {
    const video = document.getElementById('main-video');
    if (!video || !video.naturalWidth || !video.naturalHeight) return;

    const rect = video.getBoundingClientRect();
    const videoWrapper = document.getElementById('video-wrapper');
    const parentRect = videoWrapper.getBoundingClientRect();

    // 영상 DOM vs 실제 영상 비율
    const scaleX = rect.width / video.naturalWidth;
    const scaleY = rect.height / video.naturalHeight;
    const scale = Math.min(scaleX, scaleY);

    // object-fit: contain 여백 보정
    const xOffset = (rect.width - video.naturalWidth * scale) / 2;
    const yOffset = (rect.height - video.naturalHeight * scale) / 2;

    Object.keys(zoneCoords).forEach(zoneId => {
        const overlay = document.getElementById(`zone-${zoneId}`);
        if (!overlay) return;

        const coords = zoneCoords[zoneId];

        // 좌표를 overlay 내부 좌표(px)로 변환 (video 영역 내 상대 좌표)
        const polygonPoints = coords.map(([x, y]) => {
            const px = x * scale + xOffset;
            const py = y * scale + yOffset;
            return `${px}px ${py}px`;
        }).join(', ');

        // clip-path 적용
        overlay.style.clipPath = `polygon(${polygonPoints})`;
        
        // overlay 위치를 videoWrapper 대비 상대 좌표로 맞춤
        overlay.style.position = 'absolute';
        overlay.style.top = `${rect.top - parentRect.top}px`;
        overlay.style.left = `${rect.left - parentRect.left}px`;
        overlay.style.width = `${rect.width}px`;
        overlay.style.height = `${rect.height}px`;
        overlay.style.display = (currentCamera === 'main') ? 'block' : 'none';
        overlay.style.background = 'rgba(0, 255, 136, 0.3)';
        overlay.style.cursor = 'pointer';
    });
}

// MAIN 화면 overlay 표시/숨김
function toggleZoneOverlays(show) {
    Object.keys(zoneCoords).forEach(zoneId => {
        const overlay = document.getElementById(`zone-${zoneId}`);
        if (!overlay) return;
        overlay.style.display = (show && currentCamera === 'main') ? 'block' : 'none';
    });
}

function switchCamera(cameraId) {
    // 이전 탭 비활성화
    const prevTab = document.getElementById(`tab-${currentCamera}`);
    if (prevTab) prevTab.classList.remove('active');

    // 새 탭 활성화
    document.getElementById(`tab-${cameraId}`).classList.add('active');

    // 제목 업데이트
    const title = cameraId === 'main'
        ? '📹 MAIN VIEW - 전체 화면 (구역 클릭 가능)'
        : `📹 PUBLIC CCTV ${cameraId} - CAM_0${cameraId}`;
    document.getElementById('current-camera-title').textContent = title;

    // 기존 <img> src만 교체
    const videoElement = document.getElementById('main-video');
    if (videoElement) {
        videoElement.src = `/cam1/camera/${cameraId}`;
        videoElement.onload = () => calculateZonePositions();
        videoElement.onerror = () => handleVideoError(cameraId);
    }

    // MAIN 화면일 때만 오버레이 표시
    toggleZoneOverlays(cameraId === 'main');

    currentCamera = cameraId;
}

// 🔥 구역 팝업 열기
function openZonePopup(zoneId) {
    const popup = document.getElementById('zone-popup');
    const popupTitle = document.getElementById('popup-title');
    const popupVideo = document.getElementById('popup-video');
    
    popupTitle.textContent = `📹 ZONE ${zoneId} - 상세 모니터링`;
    popupVideo.src = `/cam1/camera/${zoneId}`;
    popup.style.display = 'flex';
    
    console.log(`ZONE ${zoneId} 팝업 열림`);
}

// 🔥 구역 팝업 닫기
function closeZonePopup() {
    const popup = document.getElementById('zone-popup');
    const popupVideo = document.getElementById('popup-video');
    
    popup.style.display = 'none';
    popupVideo.src = ''; // 영상 스트림 정지
}

// 🔥 위험 구역 자동 팝업 열기
function openDangerZonesPopup(dangerZoneIds) {
    const popup = document.getElementById('danger-zones-popup');
    const container = document.getElementById('danger-zones-container');
    
    // 위험 구역들을 왼쪽부터 정렬
    const sortedZones = dangerZoneIds.sort((a, b) => a - b);
    
    // 위험 구역 아이템들 생성
    container.innerHTML = sortedZones.map(zoneId => `
        <div class="danger-zone-item">
            <div class="danger-warning">위험!</div>
            <div class="danger-zone-header">
                <div class="danger-zone-title">🚨 ZONE ${zoneId}</div>
                <div class="danger-zone-count" id="danger-zone-${zoneId}-count">0명</div>
            </div>
            <img class="danger-zone-video" 
                    src="/cam1/camera/${zoneId}" 
                    id="danger-zone-${zoneId}-video"
                    onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMDAwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iI2ZmNDQ0NCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZFRUQgRVJST1I8L3RleHQ+PC9zdmc+'">
        </div>
    `).join('');
    
    // 각 위험 구역의 카운트 업데이트
    sortedZones.forEach(zoneId => {
        const countElement = document.getElementById(`danger-zone-${zoneId}-count`);
        const statElement = document.getElementById(`zone-${zoneId}-count`);
        if (countElement && statElement) {
            countElement.textContent = statElement.textContent;
        }
    });
    
    popup.style.display = 'flex';
    dangerPopupOpen = true;
    
    console.log(`위험 구역 자동 팝업 열림: ZONE ${sortedZones.join(', ')}`);
}

// 🔥 위험 구역 자동 팝업 닫기
function closeDangerZonesPopup() {
    const popup = document.getElementById('danger-zones-popup');
    const container = document.getElementById('danger-zones-container');
    
    popup.style.display = 'none';
    container.innerHTML = '';
    dangerPopupOpen = false;
    
    console.log('위험 구역 자동 팝업 닫힘');
}

// 🔥 ESC 키로 팝업 닫기
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeZonePopup();
        closeDangerZonesPopup();
    }
});

// 🔥 팝업 바깥 클릭으로 닫기
document.getElementById('zone-popup').addEventListener('click', function(event) {
    if (event.target === this) {
        closeZonePopup();
    }
});

// 🔥 위험 구역 팝업 바깥 클릭으로 닫기
document.getElementById('danger-zones-popup').addEventListener('click', function(event) {
    if (event.target === this) {
        closeDangerZonesPopup();
    }
});

function updateFrameInfo(cameraId) {
    if (cameraId === 'main') {
        document.getElementById('current-frame').textContent = 'Live - MAIN';
    } else {
        document.getElementById('current-frame').textContent = `Live - CAM_0${cameraId}`;
    }
}

function handleVideoError(cameraId) {
    const feedElement = document.getElementById('main-camera-feed');
    const errorMsg = cameraId === 'main' ? 'MAIN VIEW' : `카메라 ${cameraId}`;
    feedElement.innerHTML = `<div style="color:#ff4444;">📹 ${errorMsg} 연결 실패</div>`;
}

// 🔥 위험 구역 추적 변수
let dangerZones = new Set();
let dangerPopupOpen = false;

// 🔥 구역별 객체 탐지 결과 업데이트
async function fetchZoneDetections() {
    try {
        const response = await fetch('/cam1/api/zone_detections');
        const data = await response.json();

        const currentDangerZones = new Set();

        // 백엔드에서 내려준 임계값 동적으로 받기
        const warningThreshold = data.thresholds.warning;
        const dangerThreshold = data.thresholds.danger;

        // 구역별 탐지 인원 수 업데이트 및 상태 스타일 반영
        for (let zoneId = 1; zoneId <= 4; zoneId++) {
            const count = data.zones[zoneId] || 0;
            const countElement = document.getElementById(`zone-${zoneId}-count`);
            const statElement = document.getElementById(`zone-${zoneId}-stat`);

            if (countElement) {
                countElement.textContent = `${count}명`;
            }
            if (statElement) {
                statElement.className = 'stat-item zone-stat';
                if (count >= dangerThreshold) {
                    statElement.classList.add('danger');
                    currentDangerZones.add(zoneId);
                } else if (count >= warningThreshold) {
                    statElement.classList.add('warning');
                } else {
                    statElement.classList.add('safe');
                }
            }
        }

        // 위험 구역 자동 팝업 처리
        if (currentDangerZones.size > 0) {
            if (!dangerPopupOpen || !arraysEqual(Array.from(dangerZones), Array.from(currentDangerZones))) {
                openDangerZonesPopup(Array.from(currentDangerZones));
            }
        } else if (dangerPopupOpen) {
            closeDangerZonesPopup();
        }

        dangerZones = currentDangerZones;

        // 총 탐지 인원 업데이트
        const totalElement = document.getElementById('total-detections');
        if (totalElement) {
            totalElement.textContent = `${data.total_detections}명`;
        }

    } catch (e) {
        console.log('구역 탐지 데이터 대기중...');
    }
}

// 🔥 배열 비교 함수
function arraysEqual(a, b) {
    if (a.length !== b.length) return false;
    const sortedA = [...a].sort();
    const sortedB = [...b].sort();
    return sortedA.every((val, index) => val === sortedB[index]);
}

// API 데이터 가져오기 (기존 호환성 유지)
async function fetchStats() {
    try {
        const response = await fetch('/cam1/api/detection_stats');
        const data = await response.json();
        document.getElementById('active-cameras').textContent = data.active_cameras || 0;
    } catch (e) { console.log('통계 대기중...'); }
}

async function fetchAlerts() {
    try {
        const response = await fetch('/cam1/api/alerts');
        const data = await response.json();
        const alertsContainer = document.getElementById('alerts-container');
        if (data.alerts && data.alerts.length > 0) {
            alertsContainer.innerHTML = data.alerts.map(alert =>
                `<div class="alert-item">${alert}</div>`).join('');
        }
    } catch (e) { console.log('알림 대기중...'); }
}

function toggleCamera(cameraId) { console.log(`카메라 ${cameraId} 토글`); }
function refreshAll() { 
    fetchStats(); 
    fetchAlerts(); 
    fetchZoneDetections();
    switchCamera(currentCamera);
}

// 주기적 업데이트
setInterval(updateTime, 1000);
setInterval(fetchStats, 5000);
setInterval(fetchAlerts, 1000);
setInterval(fetchZoneDetections, 1000); 

updateTime();
fetchStats();
fetchAlerts();
fetchZoneDetections(); // 🔥 초기 구역 탐지 데이터 로드
switchCamera('main');