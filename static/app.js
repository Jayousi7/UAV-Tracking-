class DroneTracker {
    constructor() {
        this.ws = null;
        this.videoCanvas = document.getElementById('videoCanvas');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        this.videoCtx = this.videoCanvas.getContext('2d');
        this.overlayCtx = this.overlayCanvas.getContext('2d');

        this.tracks = [];
        this.selectedIds = new Set();
        this.prevTrackIds = new Set();
        this.videoW = 1280;
        this.videoH = 720;
        this.fps = 0;
        this.connected = false;

        this.cameraType = 'RGB';
        this.modelSize = 'n';
        this.trackerType = 'ocsort';
        this.targetFps = 15;
        this.threshold = 0.3;

        this.trailHistory = {};
        this.maxTrailLength = 40;

        this.setupCanvas();
        this.setupControls();
        this.connect();
        this.updateClock();
    }

    setupCanvas() {
        const container = document.getElementById('feedContainer');
        const resize = () => {
            const rect = container.getBoundingClientRect();
            const w = rect.width;
            const h = rect.height;
            this.videoCanvas.width = w;
            this.videoCanvas.height = h;
            this.overlayCanvas.width = w;
            this.overlayCanvas.height = h;
            this.canvasW = w;
            this.canvasH = h;
        };
        resize();
        new ResizeObserver(resize).observe(container);

        this.overlayCanvas.addEventListener('click', (e) => this.handleClick(e));
    }

    setupControls() {
        document.querySelectorAll('#cameraToggle .toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('#cameraToggle .toggle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.cameraType = btn.dataset.value;
            });
        });

        document.getElementById('trackerSelect').addEventListener('change', (e) => {
            this.trackerType = e.target.value;
        });

        document.getElementById('modelSelect').addEventListener('change', (e) => {
            this.modelSize = e.target.value;
        });

        document.getElementById('fpsSelect').addEventListener('change', (e) => {
            this.targetFps = parseInt(e.target.value, 10);
        });

        document.getElementById('btnApply').addEventListener('click', () => this.applyConfig());
        document.getElementById('btnPause').addEventListener('click', () => this.togglePause());
        document.getElementById('btnRestart').addEventListener('click', () => this.restart());
        document.getElementById('btnCapture').addEventListener('click', () => this.captureScreenshot());
        document.getElementById('videoUpload').addEventListener('change', (e) => this.handleVideoUpload(e));

        const slider = document.getElementById('threshSlider');
        slider.addEventListener('input', (e) => {
            this.threshold = parseFloat(e.target.value);
            document.getElementById('threshValue').textContent = this.threshold.toFixed(2);
        });
    }

    getModelKey() {
        const prefix = this.modelSize === 'DT' ? 'DT' : `yolo26${this.modelSize}`;
        return `${prefix}_${this.cameraType}`;
    }

    connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws`);

        this.ws.onopen = () => {
            this.connected = true;
            this.setStatus('online', 'SYSTEM ONLINE');
            this.setBadge('live', 'LIVE');
            this.addLog('System connected', 'acquired');
            this.updateFooter('Tracking operational');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'frame') this.handleFrame(data);
            else if (data.type === 'error') this.addLog(data.message, 'lost');
            else if (data.type === 'waiting') {
                this.addLog('Awaiting video upload...', 'highlight');
                this.setBadge('standby', 'WAITING');
            }
            else if (data.type === 'video_loaded') {
                this.addLog('Video stream starting...', 'acquired');
                this.setBadge('live', 'LIVE');
            }
            else if (data.type === 'video_info') {
                this.addLog(`Video info: ${data.width}x${data.height} @ ${data.fps}fps`, 'highlight');
            }
        };

        this.ws.onclose = () => {
            this.connected = false;
            this.setStatus('offline', 'OFFLINE');
            this.setBadge('', 'STANDBY');
            this.addLog('Connection lost', 'lost');
            setTimeout(() => this.connect(), 2000);
        };
    }

    handleFrame(data) {
        const img = new Image();
        img.onload = () => {
            if (data.videoW) this.videoW = data.videoW;
            if (data.videoH) this.videoH = data.videoH;
            
            this.videoCtx.drawImage(img, 0, 0, this.canvasW, this.canvasH);
            this.tracks = data.tracks;
            this.updateTrails(data.tracks);
            this.drawOverlay();
            this.updateStats(data);
            this.trackChanges(data.tracks);
        };
        img.src = 'data:image/jpeg;base64,' + data.frame;

        document.getElementById('overlayModel').textContent = data.model;
        document.getElementById('overlayTracker').textContent = data.tracker.toUpperCase();

        if (data.totalFrames > 0) {
            const pct = (data.frameNum / data.totalFrames) * 100;
            document.getElementById('progressFill').style.width = pct + '%';
        }
    }

    drawOverlay() {
        const ctx = this.overlayCtx;
        ctx.clearRect(0, 0, this.canvasW, this.canvasH);

        const videoW = this.videoW || 1280;
        const videoH = this.videoH || 720;
        const scaleX = this.canvasW / videoW;
        const scaleY = this.canvasH / videoH;

        // Draw trails for selected targets
        for (const t of this.tracks) {
            if (!this.selectedIds.has(t.id)) continue;
            const trail = this.trailHistory[t.id];
            if (!trail || trail.length < 2) continue;
            ctx.beginPath();
            for (let i = 0; i < trail.length; i++) {
                const x = trail[i].x * scaleX;
                const y = trail[i].y * scaleY;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = 'rgba(0, 229, 255, 0.5)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        for (const t of this.tracks) {
            const x1 = t.x1 * scaleX;
            const y1 = t.y1 * scaleY;
            const x2 = t.x2 * scaleX;
            const y2 = t.y2 * scaleY;
            const w = x2 - x1;
            const h = y2 - y1;
            const selected = this.selectedIds.has(t.id);
            const clsLabel = t.cls || '?';

            if (selected) {
                ctx.shadowColor = '#00ff41';
                ctx.shadowBlur = 12;
                ctx.strokeStyle = '#00ff41';
                ctx.lineWidth = 2.5;
                ctx.strokeRect(x1, y1, w, h);

                ctx.shadowBlur = 0;
                ctx.fillStyle = 'rgba(0, 255, 65, 0.08)';
                ctx.fillRect(x1, y1, w, h);

                const label = `TGT-${String(t.id).padStart(3, '0')} [${clsLabel}]`;
                ctx.font = '11px "Share Tech Mono"';
                const tw = ctx.measureText(label).width;
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(x1, y1 - 16, tw + 8, 16);
                ctx.fillStyle = '#00ff41';
                ctx.fillText(label, x1 + 4, y1 - 4);

                const cornerLen = Math.min(w, h) * 0.25;
                ctx.strokeStyle = '#00e5ff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x1, y1 + cornerLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cornerLen, y1);
                ctx.moveTo(x2 - cornerLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cornerLen);
                ctx.moveTo(x2, y2 - cornerLen); ctx.lineTo(x2, y2); ctx.lineTo(x2 - cornerLen, y2);
                ctx.moveTo(x1 + cornerLen, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - cornerLen);
                ctx.stroke();
            } else {
                ctx.shadowBlur = 0;
                ctx.strokeStyle = 'rgba(0, 255, 65, 0.2)';
                ctx.lineWidth = 1;
                ctx.strokeRect(x1, y1, w, h);

                ctx.font = '9px "Share Tech Mono"';
                ctx.fillStyle = 'rgba(0, 255, 65, 0.5)';
                ctx.fillText(`${t.id} ${clsLabel}`, x1 + 2, y1 - 3);
            }
        }
    }

    handleClick(e) {
        const rect = this.overlayCanvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (this.canvasW / rect.width);
        const my = (e.clientY - rect.top) * (this.canvasH / rect.height);

        const videoW = this.videoW || 1280;
        const videoH = this.videoH || 720;
        const scaleX = this.canvasW / videoW;
        const scaleY = this.canvasH / videoH;

        let clicked = null;
        let minArea = Infinity;

        for (const t of this.tracks) {
            const x1 = t.x1 * scaleX;
            const y1 = t.y1 * scaleY;
            const x2 = t.x2 * scaleX;
            const y2 = t.y2 * scaleY;

            if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
                const area = (x2 - x1) * (y2 - y1);
                if (area < minArea) {
                    minArea = area;
                    clicked = t;
                }
            }
        }

        if (clicked) {
            if (this.selectedIds.has(clicked.id)) {
                this.selectedIds.delete(clicked.id);
                this.addLog(`TGT-${clicked.id} deselected`, '');
            } else {
                this.selectedIds.add(clicked.id);
                this.addLog(`TGT-${clicked.id} highlighted`, 'highlight');
            }
            document.getElementById('statSelected').textContent = this.selectedIds.size;
            this.drawOverlay();
        }
    }

    trackChanges(tracks) {
        const currentIds = new Set(tracks.map(t => t.id));

        for (const id of currentIds) {
            if (!this.prevTrackIds.has(id)) {
                this.addLog(`TGT-${id} acquired`, 'acquired');
            }
        }

        for (const id of this.prevTrackIds) {
            if (!currentIds.has(id)) {
                this.addLog(`TGT-${id} lost`, 'lost');
                // Removed: this.selectedIds.delete(id); so it highlights again if it comes back
            }
        }

        this.prevTrackIds = currentIds;
    }

    updateTrails(tracks) {
        const activeIds = new Set();
        for (const t of tracks) {
            activeIds.add(t.id);
            const cx = (t.x1 + t.x2) / 2;
            const cy = (t.y1 + t.y2) / 2;
            if (!this.trailHistory[t.id]) this.trailHistory[t.id] = [];
            this.trailHistory[t.id].push({ x: cx, y: cy });
            if (this.trailHistory[t.id].length > this.maxTrailLength) {
                this.trailHistory[t.id].shift();
            }
        }
        for (const id in this.trailHistory) {
            if (!activeIds.has(parseInt(id))) delete this.trailHistory[id];
        }
    }

    updateStats(data) {
        document.getElementById('statFps').textContent = data.fps;
        document.getElementById('statInference').textContent = data.inferenceMs || '--';
        const persons = data.tracks.filter(t => t.cls === 'Person').length;
        const vehicles = data.tracks.filter(t => t.cls === 'Car').length;
        document.getElementById('statPersons').textContent = persons;
        document.getElementById('statVehicles').textContent = vehicles;
        document.getElementById('statSelected').textContent = this.selectedIds.size;
        document.getElementById('statFrame').textContent = data.frameNum;
    }

    applyConfig() {
        const model = this.getModelKey();
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'config',
                model: model,
                tracker: this.trackerType,
                fps: this.targetFps,
                threshold: this.threshold
            }));
            this.selectedIds.clear();
            this.prevTrackIds.clear();
            this.trailHistory = {};
            this.addLog(`Config: ${model} + ${this.trackerType.toUpperCase()} @ ${this.targetFps}FPS | Thresh: ${this.threshold}`, 'highlight');
            this.updateFooter(`Deployed: ${model} | ${this.trackerType.toUpperCase()}`);
        }
    }

    async handleVideoUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const status = document.getElementById('uploadStatus');
        status.textContent = 'UPLOADING...';
        this.addLog(`Uploading ${file.name}...`, 'highlight');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok && data.status === 'ok') {
                status.textContent = `LOADED: ${data.video.filename}`;
                this.addLog(`Video ready: ${data.video.filename}`, 'acquired');
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'new_video' }));
                }
            } else {
                status.textContent = 'UPLOAD FAILED';
                this.addLog(`Upload error: ${data.error || 'Unknown error'}`, 'lost');
            }
        } catch (err) {
            status.textContent = 'UPLOAD FAILED';
            this.addLog('Upload failed', 'lost');
        }
        e.target.value = '';
    }

    togglePause() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'pause' }));
            const btn = document.getElementById('btnPause');
            const isPaused = btn.textContent === 'RESUME';
            btn.textContent = isPaused ? 'PAUSE' : 'RESUME';
            this.setBadge(isPaused ? 'live' : '', isPaused ? 'LIVE' : 'PAUSED');
        }
    }

    restart() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'restart' }));
            this.selectedIds.clear();
            this.prevTrackIds.clear();
            this.trailHistory = {};
            document.getElementById('btnPause').textContent = 'PAUSE';
            this.setBadge('live', 'LIVE');
            this.addLog('Feed restarted', 'acquired');
        }
    }

    captureScreenshot() {
        const capture = document.createElement('canvas');
        capture.width = this.canvasW;
        capture.height = this.canvasH;
        const capCtx = capture.getContext('2d');
        capCtx.drawImage(this.videoCanvas, 0, 0);
        capCtx.drawImage(this.overlayCanvas, 0, 0);
        const link = document.createElement('a');
        link.download = `capture_${Date.now()}.png`;
        link.href = capture.toDataURL('image/png');
        link.click();
        this.addLog('Screenshot captured', 'acquired');
    }

    setStatus(state, text) {
        const dot = document.getElementById('statusDot');
        const txt = document.getElementById('statusText');
        dot.className = 'status-dot ' + state;
        txt.className = 'status-text ' + state;
        txt.textContent = text;
    }

    setBadge(cls, text) {
        const badge = document.getElementById('feedBadge');
        badge.className = 'feed-badge ' + cls;
        badge.textContent = text;
    }

    addLog(message, cls = '') {
        const log = document.getElementById('targetLog');
        const entry = document.createElement('div');
        entry.className = 'log-entry ' + cls;
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        entry.textContent = `[${time}] ${message}`;
        log.prepend(entry);

        while (log.children.length > 50) log.removeChild(log.lastChild);
    }

    updateFooter(msg) {
        document.getElementById('footerStatus').textContent = 'SYS: ' + msg;
    }

    updateClock() {
        const el = document.getElementById('footerTime');
        const update = () => {
            el.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
        };
        update();
        setInterval(update, 1000);
    }
}

document.addEventListener('DOMContentLoaded', () => new DroneTracker());
