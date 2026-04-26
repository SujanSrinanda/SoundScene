// Register Service Worker for PWA
if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
        navigator.serviceWorker.register("/static/service-worker.js")
            .then(reg => console.log("Service Worker Registered!", reg))
            .catch(err => console.log("Service Worker Registration Failed!", err));
    });
}

const API_URL = '';

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
const canvasCtx = canvas.getContext('2d');

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
    try {
        const profileRes = await fetch(`${API_URL}/profile`);
        if (profileRes.ok) {
            const profile = await profileRes.json();
            if (profile.name) userName = profile.name;
            if (profile.location) userLocation = profile.location;
            
            profileNameInput.value = userName;
            profileLocationInput.value = userLocation;
            
            if(userName) statusText.textContent = `Tap to Listen, ${userName}`;
        }
    } catch(e) {
        console.error("Failed to load initial data", e);
    }
}
window.addEventListener('DOMContentLoaded', loadInitialData);

// Save Profile
saveProfileBtn.addEventListener('click', async () => {
    const newName = profileNameInput.value.trim();
    const newLocation = profileLocationInput.value;
    try {
        await fetch(`${API_URL}/profile`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: newName, location: newLocation})
        });
        userName = newName;
        userLocation = newLocation;
        showToast("Profile updated!");
        profileModal.classList.remove('active');
        if(!isListening && userName) statusText.textContent = `Tap to Listen, ${userName}`;
    } catch(e) {
        showToast("Failed to save profile");
    }
});

// --- Wizard Navigation ---
function goToStep(stepId) {
    wizardSteps.forEach(s => s.classList.remove('active'));
    document.getElementById(`step-${stepId}`).classList.add('active');
}

// Step 1 -> Stop Recording
startRecordBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = () => {
            const recordedAudioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioPlayback.src = URL.createObjectURL(recordedAudioBlob);
            
            const reader = new FileReader();
            reader.readAsDataURL(recordedAudioBlob);
            reader.onloadend = () => { recordedAudioB64 = reader.result; };
            
            goToStep('playback');
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        startRecordBtn.classList.add('hidden');
        stopRecordBtn.classList.remove('hidden');
        recordPulse.classList.remove('hidden');
    } catch(err) {
        showToast("Mic access denied");
    }
});

stopRecordBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        startRecordBtn.classList.remove('hidden');
        stopRecordBtn.classList.add('hidden');
        recordPulse.classList.add('hidden');
    }
});

rerecordBtn.addEventListener('click', () => {
    goToStep('record');
    recordedAudioB64 = "";
    audioPlayback.src = "";
});

keepAudioBtn.addEventListener('click', () => goToStep('metadata'));

previewBtn.addEventListener('click', () => {
    if (!customIdInput.value || !customDescInput.value) {
        showToast("Please fill all fields");
        return;
    }
    
    // Render Preview
    const icon = customIconInput.value || '🔊';
    const name = customIdInput.value;
    const desc = customDescInput.value;
    
    previewCardRender.innerHTML = `
        <div class="preview-icon-box">${icon}</div>
        <div>
            <h5 style="font-weight:700; font-size:16px; margin:0">${name}</h5>
            <p style="font-size:12px; color:var(--text-secondary); margin:4px 0 0 0">${desc}</p>
        </div>
    `;
    
    goToStep('preview');
});

backMetadataBtn.addEventListener('click', () => goToStep('metadata'));

// Save Custom Sound
saveSoundBtn.addEventListener('click', async () => {
    const label = customIdInput.value.trim().toLowerCase().replace(/\s+/g, '_');
    const icon = customIconInput.value.trim() || '🔊';
    const desc = customDescInput.value.trim();
    const category = customCategoryInput.value;
    
    try {
        const res = await fetch(`${API_URL}/custom_sound`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                label, icon, description: desc, category,
                audio_b64: recordedAudioB64
            })
        });
        
        if (res.ok) {
            showToast("Custom sound trained successfully!");
            settingsModal.classList.remove('active');
            
            // Reset Wizard
            goToStep('record');
            customIdInput.value = ''; customIconInput.value = ''; customDescInput.value = '';
            recordedAudioB64 = "";
            
            // Simulate detection of this new sound as a demo
            setTimeout(() => triggerCustomDetection(label), 1000);
        }
    } catch(e) {
        showToast("Training failed");
    }
});

async function triggerCustomDetection(label) {
    try {
        const response = await fetch(`${API_URL}/detect_custom`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label })
        });
        if (response.ok) {
            const data = await response.json();
            triggerAlert(data);
        }
    } catch(e) {}
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
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(analyser);
        analyser.fftSize = 256;
        
        isListening = true;
        resetUI();
        drawVisualizer();
        inferenceIntervalId = setInterval(runInference, 3000);
        
    } catch (err) {
        showToast("Mic error");
    }
}

function stopListening() {
    isListening = false;
    if (inferenceIntervalId) clearInterval(inferenceIntervalId);
    if (animationId) cancelAnimationFrame(animationId);
    if (audioContext && audioContext.state !== 'closed') audioContext.close();
    
    appContainer.classList.remove('is-detecting');
    body.className = 'mode-idle';
    statusText.textContent = userName ? `Tap to Listen, ${userName}` : "Tap to Listen";
}

function drawVisualizer() {
    if (!isListening) return;
    animationId = requestAnimationFrame(drawVisualizer);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
}

async function runInference() {
    if (appContainer.classList.contains('is-detecting')) return;

    const requestData = {
        simulate_label: "", 
        time_of_day: getCurrentTimeOfDay(),
        location_mode: userLocation
    };

    try {
        const response = await fetch(`${API_URL}/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (response.ok) {
            const data = await response.json();
            if (data && data.sound) triggerAlert(data);
        }
    } catch (error) {}
}

async function resetApp() {
    try {
        await fetch(`${API_URL}/reset`, { method: 'POST' });
        resetUI();
        showToast("System reset successful");
    } catch(e) {
        showToast("Reset failed");
    }
}

function triggerAlert(data) {
    if(autoDismissTimeout) { clearTimeout(autoDismissTimeout); autoDismissTimeout = null; }

    let mode = 'normal';
    let color = themes.green;
    
    if (data.urgency === 'medium') { mode = 'warning'; color = themes.yellow; }
    else if (data.urgency === 'critical') { mode = 'critical'; color = themes.red; }
    else { mode = 'normal'; color = themes.green; }

    let soundLabel = data.sound.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    if(data.sound === "calling_user_name" && userName) {
        soundLabel = `Calling ${userName}`;
    }

    alertTitle.textContent = soundLabel;
    alertDesc.textContent = data.situation;
    alertEmoji.textContent = data.icon || '🔊';
    alertMatch.textContent = `${Math.round(data.confidence * 100)}% Match`;
    
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
        statusText.textContent = userName ? `Tap to Listen, ${userName}` : "Tap to Listen";
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
async function triggerDemo(simulate_label, force_urgency = "") {
    const requestData = {
        simulate_label: simulate_label,
        time_of_day: getCurrentTimeOfDay(),
        location_mode: userLocation,
        force_urgency: force_urgency
    };
    try {
        const response = await fetch(`${API_URL}/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        if (response.ok) {
            const data = await response.json();
            if (data && data.sound) triggerAlert(data);
        }
    } catch(e) {}
}

demoNormal.addEventListener('click', () => triggerDemo('dog_bark', 'normal'));
demoWarning.addEventListener('click', () => triggerDemo('door_wood_knock', 'medium'));
demoCritical.addEventListener('click', () => triggerDemo('calling_user_name', 'critical'));
