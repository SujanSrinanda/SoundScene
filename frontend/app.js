// Register Service Worker for PWA
if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
        navigator.serviceWorker.register("/static/service-worker.js")
            .then(reg => console.log("Service Worker Registered!", reg))
            .catch(err => console.log("Service Worker Registration Failed!", err));
    });
}

// Task 1: Absolute API URL (Required for Ngrok/Remote access)
const API_URL = window.location.protocol === 'https:'
    ? window.location.origin
    : 'http://127.0.0.1:8000';

let isRequesting = false; // Task 4: Prevent multiple/blocking calls

let lastErrorShown = false;
function showErrorOnce(message) {
    if (lastErrorShown) return;
    lastErrorShown = true;
    showToast(message);
    setTimeout(() => { lastErrorShown = false; }, 5000);
}

async function safeApiFetch(url, options = {}) {
    console.log('API request start', url, options.method || 'GET');
    try {
        const response = await fetch(url, options);
        console.log('API request end', url, response.status);
        if (!response.ok) {
            const text = await response.text();
            throw new Error(`API error ${response.status}: ${text}`);
        }
        const contentType = response.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
            return await response.json();
        }
        return null;
    } catch (error) {
        console.error('API fetch failed', url, error);
        showErrorOnce('Connection error');
        return null;
    }
}

// Elements
const body = document.getElementById('main-body');
const appContainer = document.getElementById('app');
const olliBot = document.getElementById('olli-bot');
const statusText = document.getElementById('status-text');

const alertEmoji = document.getElementById('emoji');
const alertTitle = document.getElementById('title');
const alertDesc = document.getElementById('desc');
const alertMatch = document.getElementById('match');
const actionBtn = document.getElementById('action-btn');

const canvas = document.getElementById('audio-visualizer');
const canvasCtx = canvas ? canvas.getContext('2d') : null;

const demoNormal = document.getElementById('demo-normal');
const demoWarning = document.getElementById('demo-warning');
const demoCritical = document.getElementById('demo-critical');
const btnReset = document.getElementById('btn-reset');

// Settings Elements
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeSettings = document.getElementById('close-settings');
const autoDismissToggle = document.getElementById('auto-dismiss-toggle');
const micSensitivity = document.getElementById('mic-sensitivity');

// Wizard Elements
const wizardSteps = document.querySelectorAll('.wizard-step');
const navItems = document.querySelectorAll('.nav-item');
const settingsViews = document.querySelectorAll('.settings-view');

const startRecordBtn = document.getElementById('start-record-btn');
const stopRecordBtn = document.getElementById('stop-record-btn');
const audioPlayback = document.getElementById('audio-playback');
const recordPulse = document.getElementById('record-pulse');

const keepAudioBtn = document.getElementById('keep-audio-btn');
const rerecordBtn = document.getElementById('rerecord-btn');

const previewBtn = document.getElementById('preview-btn');
const backMetadataBtn = document.getElementById('back-metadata-btn');
const saveSoundBtn = document.getElementById('save-sound-btn');

const customIdInput = document.getElementById('custom-id');
const customCategoryInput = document.getElementById('custom-category');
const customDescInput = document.getElementById('custom-desc');
const customIconInput = document.getElementById('custom-icon');
const previewCardRender = document.getElementById('preview-card-render');

// Profile Elements
const profileBtn = document.getElementById('profile-btn');
const profileModal = document.getElementById('profile-modal');
const closeProfile = document.getElementById('close-profile');
const profileNameInput = document.getElementById('profile-name');
const profileLocationInput = document.getElementById('profile-location');
const saveProfileBtn = document.getElementById('save-profile-btn');

// Toast Container
const toastContainer = document.getElementById('toast-container');

// State
let isListening = false;
let audioContext;
let analyser;
let microphone;
let mediaStream;
let animationId;
let inferenceIntervalId;

let userName = "";
let userLocation = "home";
let autoDismissTimeout = null;

let mediaRecorder;
let audioChunks = [];
let recordedAudioB64 = "";

// Theme Colors
const themes = {
    green: '#22C55E',
    yellow: '#EAB308',
    red: '#EF4444',
    listening: '#38bdf8',
    idle: '#64748b'
};

// --- Toast System ---
function showToast(message) {
    toastContainer.innerHTML = ''; // Clear previous messages (Task 3: No Spam)
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-20px)';
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}

// --- Tab Switching Logic (Scoped to Modal) ---
document.querySelectorAll('.modal-content').forEach(modal => {
    const navItems = modal.querySelectorAll('.nav-item');
    const views = modal.querySelectorAll('.settings-view');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const section = item.getAttribute('data-section');
            if (!section) return;

            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            views.forEach(v => v.classList.remove('active'));
            const targetView = modal.querySelector(`#view-${section}`) || document.getElementById(`view-${section}`);
            if (targetView) targetView.classList.add('active');
        });
    });
});

// --- Modal Logic ---
settingsBtn.addEventListener('click', () => settingsModal.classList.add('active'));
closeSettings.addEventListener('click', () => settingsModal.classList.remove('active'));
settingsModal.addEventListener('click', (e) => {
    if(e.target === settingsModal) settingsModal.classList.remove('active');
});

profileBtn.addEventListener('click', () => profileModal.classList.add('active'));
closeProfile.addEventListener('click', () => profileModal.classList.remove('active'));
profileModal.addEventListener('click', (e) => {
    if(e.target === profileModal) profileModal.classList.remove('active');
});

// Helper: Time of Day
function getCurrentTimeOfDay() {
    const hour = new Date().getHours();
    if (hour >= 6 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 18) return 'afternoon';
    return 'night';
}

// Load Initial Data
async function loadInitialData() {
    const profile = await safeApiFetch(`${API_URL}/profile`);
    if (!profile) return;

    if (profile.name) userName = profile.name;
    if (profile.location) userLocation = profile.location;

    profileNameInput.value = userName;
    profileLocationInput.value = userLocation;

    if (userName) statusText.textContent = `Tap here to listen, ${userName}`;
}
window.addEventListener('DOMContentLoaded', loadInitialData);

// Save Profile
saveProfileBtn.addEventListener('click', async () => {
    if (saveProfileBtn.disabled) return; 

    const newName = profileNameInput.value.trim();
    const newLocation = profileLocationInput.value;

    if (!newName) {
        showToast("Display name is required");
        return;
    }

    saveProfileBtn.disabled = true;
    const originalText = saveProfileBtn.textContent;
    saveProfileBtn.textContent = "Saving...";

    try {
        const result = await safeApiFetch(`${API_URL}/save-profile`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ displayName: newName, location: newLocation })
        });

        if (!result) return;
        if (result.success) {
            userName = newName;
            userLocation = newLocation;
            showToast('Profile updated!');
            profileModal.classList.remove('active');
            if (!isListening && userName) statusText.textContent = `Tap here to listen, ${userName}`;
        } else {
            showToast(result.message || 'Failed to save profile');
        }
    } catch (e) {
        console.error('Fetch error:', e);
        showErrorOnce('Connection error. Please check your backend.');
    } finally {
        saveProfileBtn.disabled = false;
        saveProfileBtn.textContent = originalText;
    }
});

// --- Wizard Navigation ---
function goToStep(stepId) {
    wizardSteps.forEach(s => s.classList.remove('active'));
    document.getElementById(`step-${stepId}`).classList.add('active');
}

// Step 1: Start Recording
startRecordBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/wav' });
            audioPlayback.src = URL.createObjectURL(blob);
            
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = () => { recordedAudioB64 = reader.result; };
            
            goToStep('playback');
            stream.getTracks().forEach(t => t.stop());
        };
        
        mediaRecorder.start();
        startRecordBtn.classList.add('hidden');
        stopRecordBtn.classList.remove('hidden');
        recordPulse.classList.remove('hidden');
    } catch(err) {
        showToast("Microphone access denied");
    }
});

stopRecordBtn.addEventListener('click', () => {
    if (mediaRecorder) mediaRecorder.stop();
    startRecordBtn.classList.remove('hidden');
    stopRecordBtn.classList.add('hidden');
    recordPulse.classList.add('hidden');
});

rerecordBtn.addEventListener('click', () => goToStep('record'));
keepAudioBtn.addEventListener('click', () => goToStep('metadata'));

previewBtn.addEventListener('click', () => {
    if (!customIdInput.value || !customDescInput.value) {
        showToast("Please fill in the sound name and description");
        return;
    }
    previewCardRender.innerHTML = `
        <div class="preview-icon-box">${customIconInput.value || '🔊'}</div>
        <div>
            <h5 style="font-weight:700; font-size:16px; margin:0">${customIdInput.value}</h5>
            <p style="font-size:12px; color:var(--text-secondary); margin:4px 0 0 0">${customDescInput.value}</p>
        </div>
    `;
    goToStep('preview');
});

// Step 4: Save & Train (Fingerprint)
saveSoundBtn.addEventListener('click', async () => {
    saveSoundBtn.disabled = true;
    saveSoundBtn.textContent = "Fingerprinting...";
    
    const payload = {
        label: customIdInput.value.trim().toLowerCase().replace(/\s+/g, '_'),
        icon: customIconInput.value || '🔊',
        description: customDescInput.value,
        category: customCategoryInput.value,
        audio_b64: recordedAudioB64
    };

    try {
        const result = await safeApiFetch(`${API_URL}/custom_sound`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        if (!result) return;

        if (result.success) {
            showToast('Sound fingerprinted successfully');
            settingsModal.classList.remove('active');
            goToStep('record');
            customIdInput.value = '';
            customDescInput.value = '';
        } else {
            showToast('Training failed: ' + (result.message || 'unknown error'));
        }
    } catch (e) {
        console.error('Custom sound save error', e);
        showErrorOnce('Connection error during training');
    } finally {
        saveSoundBtn.disabled = false;
        saveSoundBtn.textContent = "Save & Train";
    }
});

async function triggerCustomDetection(label) {
    const data = await safeApiFetch(`${API_URL}/detect_custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label })
    });
    if (data && data.sound) {
        triggerAlert(data);
    }
}

// --- Main Audio Logic ---
olliBot.addEventListener('click', async () => {
    if (isListening) {
        stopListening();
    } else {
        await startListening();
    }
});

async function startListening() {
    try {
        console.log('DEBUG: [Frontend] Requesting microphone permission...');
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log('DEBUG: [Frontend] Microphone permission granted, stream active:', !!stream);
        mediaStream = stream;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(analyser);
        analyser.fftSize = 256;
        
        isListening = true;
        resetUI();
        drawVisualizer();
        inferenceIntervalId = setInterval(runInference, 5000);
        console.log('DEBUG: [Frontend] Listening started, inference interval set');
        
    } catch (err) {
        console.error('DEBUG: [Frontend] Microphone error:', err);
        showToast(err.name === 'NotAllowedError' ? 'Mic permission denied' : 'Mic error — check your device');
    }
}

function stopListening() {
    isListening = false;
    if (inferenceIntervalId) clearInterval(inferenceIntervalId);
    if (animationId) cancelAnimationFrame(animationId);
    if (audioContext && audioContext.state !== 'closed') audioContext.close();
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    appContainer.classList.remove('is-detecting');
    body.className = 'mode-idle';
    statusText.textContent = userName ? `Tap here to listen, ${userName}` : "Tap here to listen";
}

function drawVisualizer() {
    if (!isListening) return;
    animationId = requestAnimationFrame(drawVisualizer);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    
    // Log audio levels occasionally
    if (Math.random() < 0.01) { // Log ~1% of frames
        const avgLevel = dataArray.reduce((a, b) => a + b) / bufferLength;
        console.log('DEBUG: [Frontend] Audio level:', avgLevel.toFixed(2));
    }
}

async function recordAudio(durationMs) {
    return new Promise((resolve) => {
        console.log('DEBUG: [Frontend] Starting audio recording for', durationMs, 'ms');
        
        if (!audioContext || !microphone) {
            console.error('DEBUG: [Frontend] Audio context or microphone not initialized');
            resolve(null);
            return;
        }

        const sampleRate = audioContext.sampleRate;
        const numSamples = Math.floor((durationMs / 1000) * sampleRate);
        const audioBuffer = new Float32Array(numSamples);
        let sampleIndex = 0;

        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (event) => {
            const inputBuffer = event.inputBuffer.getChannelData(0);
            for (let i = 0; i < inputBuffer.length && sampleIndex < numSamples; i++) {
                audioBuffer[sampleIndex++] = inputBuffer[i];
            }
        };

        microphone.connect(processor);
        processor.connect(audioContext.destination);

        setTimeout(() => {
            microphone.disconnect(processor);
            processor.disconnect(audioContext.destination);
            
            console.log('DEBUG: [Frontend] Recording stopped, samples captured:', sampleIndex);
            
            // Convert Float32Array to WAV blob
            const wavBlob = float32ArrayToWav(audioBuffer, sampleRate);
            const reader = new FileReader();
            reader.onloadend = () => {
                console.log('DEBUG: [Frontend] Audio encoded to base64, length:', reader.result.length);
                resolve(reader.result);
            };
            reader.readAsDataURL(wavBlob);
        }, durationMs);
    });
}

// Helper function to convert Float32Array to WAV
function float32ArrayToWav(buffer, sampleRate) {
    const length = buffer.length;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);
    
    // PCM data
    let offset = 44;
    for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, buffer[i]));
        view.setInt16(offset, sample * 0x7FFF, true);
        offset += 2;
    }
    
    return new Blob([view], { type: 'audio/wav' });
}

async function runInference() {
    if (appContainer.classList.contains('is-detecting')) return;

    console.log('DEBUG: [Frontend] Running inference...');
    // Record 1 seconds of audio
    const audioData = await recordAudio(1000);
    
    const requestData = {
        audio_data: audioData || "",
        time_of_day: getCurrentTimeOfDay(),
        location_mode: userLocation
    };

    console.log('DEBUG: [Frontend] Sending audio to backend, data length:', requestData.audio_data.length);
    const data = await safeApiFetch(`${API_URL}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    });

    console.log('DEBUG: [Frontend] Response received:', data);
    if (data && data.sound) {
        triggerAlert(data);
    } else if (data && data.status === 'analyzing') {
        console.log('DEBUG: [Frontend] No sound detected');
        showToast('No sound detected');
    } else {
        console.log('DEBUG: [Frontend] Invalid response or error');
    }
}

async function resetApp() {
    const result = await safeApiFetch(`${API_URL}/reset`, { method: 'POST' });
    if (result && result.status === 'success') {
        resetUI();
        showToast('System reset successful');
    } else {
        showErrorOnce('Reset failed');
    }
}

function triggerAlert(data) {
    if(autoDismissTimeout) { clearTimeout(autoDismissTimeout); autoDismissTimeout = null; }

    let mode = 'normal';
    let color = themes.green;
    
    if (data.urgency === 'medium') { mode = 'warning'; color = themes.yellow; }
    else if (data.urgency === 'critical') { mode = 'critical'; color = themes.red; }
    else { mode = 'normal'; color = themes.green; }

    if (!data || !data.sound) return;

    let soundLabel = data.sound.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    if(data.sound === "calling_user_name" && userName) {
        soundLabel = `Calling ${userName}`;
    }

    alertTitle.textContent = soundLabel || 'Unknown Sound';
    alertDesc.textContent = data.situation || 'Sound detected';
    alertEmoji.textContent = data.icon || '🔊';
    alertMatch.textContent = `${Math.round((data.confidence || 0) * 100)}% Match`;
    
    if (data.is_custom) {
        alertMatch.innerHTML = `${Math.round(data.confidence * 100)}% Match <span class="custom-badge" style="background:rgba(255,255,255,0.1); padding:2px 6px; border-radius:6px; margin-left:8px; color:var(--accent-yellow)">Custom</span>`;
    }
    
    actionBtn.textContent = data.action;
    statusText.textContent = "Signal Found";
    document.documentElement.style.setProperty('--current-accent', color);
    
    appContainer.classList.add('is-detecting');
    body.className = `mode-${mode}`;

    if (autoDismissToggle.checked) {
        autoDismissTimeout = setTimeout(resetUI, 5000);
    }
}

function resetUI() {
    if(autoDismissTimeout) { clearTimeout(autoDismissTimeout); autoDismissTimeout = null; }
    if (!isListening) {
        appContainer.classList.remove('is-detecting');
        body.className = 'mode-idle';
        statusText.textContent = userName ? `Tap here to listen, ${userName}` : "Tap here to listen";
        return;
    }
    
    appContainer.classList.remove('is-detecting');
    body.className = 'mode-listening';
    statusText.textContent = userName ? `Listening, ${userName}...` : "Listening...";
    document.documentElement.style.setProperty('--current-accent', themes.listening);
}

// Buttons
actionBtn.addEventListener('click', resetUI);
btnReset.addEventListener('click', resetApp);

// Demo Buttons
const demoData = {
    dog_bark: { sound: 'dog_bark', icon: '🐕', situation: 'Dog barking nearby', action: 'Acknowledge', confidence: 0.92 },
    door_wood_knock: { sound: 'door_wood_knock', icon: '🚪', situation: 'Repeated knocking on door', action: 'Investigate', confidence: 0.88 },
    calling_user_name: { sound: 'calling_user_name', icon: '📢', situation: 'Someone calling urgently', action: 'Immediate Action', confidence: 0.95 }
};

async function triggerDemo(simulate_label, force_urgency = "") {
    const requestData = {
        simulate_label: simulate_label,
        time_of_day: getCurrentTimeOfDay(),
        location_mode: userLocation,
        force_urgency: force_urgency
    };
    const data = await safeApiFetch(`${API_URL}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    });
    if (data && data.sound) {
        triggerAlert(data);
        return;
    }

    // Fallback: backend not reachable — use local mock data
    const mock = { ...demoData[simulate_label], urgency: force_urgency || 'normal' };
    triggerAlert(mock);
}

demoNormal.addEventListener('click', () => triggerDemo('dog_bark', 'normal'));
demoWarning.addEventListener('click', () => triggerDemo('door_wood_knock', 'medium'));
demoCritical.addEventListener('click', () => triggerDemo('calling_user_name', 'critical'));
