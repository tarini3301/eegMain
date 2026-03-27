/**
 * NeuroAge — Brain Age Prediction Frontend
 * Multi-Model Edition
 * ==========================================
 * Handles: form building, sample loading, API calls,
 * gauge rendering, SHAP charts, importance charts,
 * recommendation display, and model selection.
 */

// ═══════════════ CONFIGURATION ═══════════════

const REGIONS = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"];
const BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"];

const FEATURE_DISPLAY = {};
const FEATURE_UNITS = {};

// ── Exact per-feature ranges from synthetic_dataset.csv ──
const FEATURE_DEFAULTS = {
    Frontal_Delta_Power:    { min: 21.0, max: 48.0, step: 0.01, placeholder: '34.18', icon: '🌊' },
    Frontal_Theta_Power:    { min: 13.0, max: 28.0, step: 0.01, placeholder: '20.18', icon: '🧘' },
    Frontal_Alpha_Power:    { min: 2.0,  max: 25.0, step: 0.01, placeholder: '13.81', icon: '⚡' },
    Frontal_Beta_Power:     { min: 5.0,  max: 16.0, step: 0.01, placeholder: '10.94', icon: '🎯' },
    Frontal_Gamma_Power:    { min: 1.5,  max: 7.5,  step: 0.01, placeholder: '4.59',  icon: '🧠' },

    Central_Delta_Power:    { min: 18.0, max: 41.0, step: 0.01, placeholder: '29.51', icon: '🌊' },
    Central_Theta_Power:    { min: 10.0, max: 24.0, step: 0.01, placeholder: '17.22', icon: '🧘' },
    Central_Alpha_Power:    { min: 0.1,  max: 26.0, step: 0.01, placeholder: '14.08', icon: '⚡' },
    Central_Beta_Power:     { min: 5.0,  max: 16.0, step: 0.01, placeholder: '10.68', icon: '🎯' },
    Central_Gamma_Power:    { min: 2.0,  max: 7.5,  step: 0.01, placeholder: '4.62',  icon: '🧠' },

    Temporal_Delta_Power:   { min: 14.0, max: 43.0, step: 0.01, placeholder: '28.71', icon: '🌊' },
    Temporal_Theta_Power:   { min: 10.0, max: 24.0, step: 0.01, placeholder: '16.81', icon: '🧘' },
    Temporal_Alpha_Power:   { min: 2.0,  max: 26.0, step: 0.01, placeholder: '13.91', icon: '⚡' },
    Temporal_Beta_Power:    { min: 5.0,  max: 17.0, step: 0.01, placeholder: '10.72', icon: '🎯' },
    Temporal_Gamma_Power:   { min: 2.5,  max: 6.5,  step: 0.01, placeholder: '4.59',  icon: '🧠' },

    Parietal_Delta_Power:   { min: 16.0, max: 43.0, step: 0.01, placeholder: '29.42', icon: '🌊' },
    Parietal_Theta_Power:   { min: 10.0, max: 24.0, step: 0.01, placeholder: '17.50', icon: '🧘' },
    Parietal_Alpha_Power:   { min: 9.0,  max: 41.0, step: 0.01, placeholder: '29.34', icon: '⚡' },
    Parietal_Beta_Power:    { min: 5.5,  max: 16.0, step: 0.01, placeholder: '10.74', icon: '🎯' },
    Parietal_Gamma_Power:   { min: 2.0,  max: 7.0,  step: 0.01, placeholder: '4.56',  icon: '🧠' },

    Occipital_Delta_Power:  { min: 15.0, max: 45.0, step: 0.01, placeholder: '29.96', icon: '🌊' },
    Occipital_Theta_Power:  { min: 8.5,  max: 24.0, step: 0.01, placeholder: '17.42', icon: '🧘' },
    Occipital_Alpha_Power:  { min: 12.0, max: 47.0, step: 0.01, placeholder: '29.08', icon: '⚡' },
    Occipital_Beta_Power:   { min: 6.5,  max: 15.0, step: 0.01, placeholder: '10.71', icon: '🎯' },
    Occipital_Gamma_Power:  { min: 2.5,  max: 7.0,  step: 0.01, placeholder: '4.65',  icon: '🧠' }
};

// Populate display names
REGIONS.forEach(region => {
    BANDS.forEach(band => {
        const fname = `${region}_${band}_Power`;
        FEATURE_DISPLAY[fname] = `${region} ${band} Power`;
        FEATURE_UNITS[fname] = 'µV²';
    });
});

// ── PDF Export (Clinical Report) ──
function exportClinicalReport() {
    const element = document.getElementById('report-container');
    const btnContainer = document.getElementById('export-btn-container');

    if (btnContainer) btnContainer.style.display = 'none';

    // Highlight the container for a clean professional look in PDF
    element.style.padding = '20px';
    element.style.background = '#0f172a';

    const opt = {
        margin:       [0.5, 0.3, 0.5, 0.3], // [top, left, bottom, right]
        filename:     `NeuroAge_Clinical_Report_${new Date().toISOString().split('T')[0]}.pdf`,
        image:        { type: 'jpeg', quality: 1.0 },
        html2canvas:  { 
            scale: 2, 
            useCORS: true, 
            logging: false,
            backgroundColor: '#0f172a',
            windowWidth: 1200 // Ensure consistent layout in PDF
        },
        jsPDF:        { unit: 'in', format: 'a4', orientation: 'portrait' },
        pagebreak:    { mode: ['avoid-all', 'css', 'legacy'] }
    };

    // Use a small timeout to let any animations settle
    setTimeout(() => {
        html2pdf().set(opt).from(element).save().then(() => {
            if (btnContainer) btnContainer.style.display = 'flex';
            element.style.padding = '0';
        }).catch(err => {
            console.error('PDF export failed:', err);
            if (btnContainer) btnContainer.style.display = 'flex';
            element.style.padding = '0';
        });
    }, 2000);
}


// ═══════════════ INITIALIZATION & STATE ═══════════════

let activeModelKey = 'ensemble';
let threeScene, threeCamera, threeRenderer, threeBrain;
let prognosisChart = null;

document.addEventListener('DOMContentLoaded', () => {
    buildFeatureInputs();
    loadModels();
    loadSampleSubjects();
    setupFormHandlers();
    initTabNavigation();
});


function buildFeatureInputs() {
    const grid = document.getElementById('features-grid');
    grid.innerHTML = '';

    for (const [fname, config] of Object.entries(FEATURE_DEFAULTS)) {
        const card = document.createElement('div');
        card.className = 'feature-input-card';
        const unit = FEATURE_UNITS[fname] ? ` (${FEATURE_UNITS[fname]})` : '';
        card.innerHTML = `
            <label for="${fname}">
                <span class="label-icon">${config.icon}</span>
                ${FEATURE_DISPLAY[fname]}${unit}
            </label>
            <div class="input-slider-group">
                <input type="number" id="${fname}" name="${fname}"
                       min="${config.min}" max="${config.max}" step="${config.step}"
                       placeholder="${config.placeholder}"
                       class="input-field" required>
                <input type="range" id="${fname}_slider"
                       min="${config.min}" max="${config.max}" step="${config.step}"
                       value="${config.placeholder}"
                       class="slider-field">
            </div>
        `;
        grid.appendChild(card);

        // Bind events for live sync & prediction
        const numInput = document.getElementById(fname);
        const sliderInput = document.getElementById(`${fname}_slider`);

        numInput.addEventListener('input', (e) => {
            sliderInput.value = e.target.value;
            triggerFastPrediction();
        });
        
        sliderInput.addEventListener('input', (e) => {
            numInput.value = e.target.value;
            triggerFastPrediction();
        });
    }
}

// ── Fast prediction polling ──
let fastPredictionTimer = null;
function triggerFastPrediction() {
    const section = document.getElementById('results-section');
    if (section.style.display === 'none' || section.style.display === '') return;
    
    if (fastPredictionTimer) clearTimeout(fastPredictionTimer);
    fastPredictionTimer = setTimeout(async () => {
        const features = {};
        for (const fname of Object.keys(FEATURE_DEFAULTS)) {
            const val = parseFloat(document.getElementById(fname).value);
            if (isNaN(val)) return;
            features[fname] = val;
        }
        
        try {
            const res = await fetch('/api/predict_fast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    features, 
                    model: activeModelKey
                }),
            });
            if (res.ok) {
                const data = await res.json();
                if (data.predicted_age) {
                    drawGauge(data.predicted_age);
                    document.getElementById('gauge-value').textContent = data.predicted_age.toFixed(1);
                    
                    const ageInput = document.getElementById('chronological_age');
                    const chronoAge = ageInput.value ? parseFloat(ageInput.value) : null;
                    if (chronoAge) {
                        const gap = data.predicted_age - chronoAge;
                        const resultMock = { brain_age_gap: gap, chronological_age: chronoAge };
                        // We do not call displayGap(resultMock) directly because it reconstructs innerHTML. 
                        // Just update the text manually to avoid flashing metrics
                        const gapValEl = document.querySelector('.gap-value');
                        const gapDescEl = document.querySelector('.gap-description');
                        if (gapValEl && gapDescEl) {
                            const sign = gap > 0 ? "+" : "";
                            gapValEl.textContent = `${sign}${gap.toFixed(1)}y`;
                            let color = 'var(--text-primary)';
                            let desc = 'Your brain age matches your chronological age.';
                            if (gap > 2) {
                                color = 'var(--amber)';
                                desc = 'Accelerated brain aging detected. Review recommendations.';
                            } else if (gap < -2) {
                                color = 'var(--green)';
                                desc = 'Healthy aging profile. Your brain appears younger.';
                            }
                            gapValEl.style.color = color;
                            gapDescEl.textContent = desc;
                        }
                    }
                }
            }
        } catch (e) {
            console.error("Fast prediction failed:", e);
        }
    }, 100);
}


// ═══════════════ API LOADERS ═══════════════

async function loadModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        availableModels = data.models || [];
        renderModelSelector();
    } catch (e) {
        console.error('Failed to load models:', e);
        document.getElementById('model-selector').innerHTML =
            '<span style="color:var(--text-muted);font-size:0.82rem;">Could not load models. Server may be offline.</span>';
    }
}

async function loadSampleSubjects() {
    try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        sampleData = data.samples || [];
        renderSampleButtons();
    } catch (e) {
        console.error('Failed to load samples:', e);
        document.getElementById('sample-buttons').innerHTML =
            '<span style="color:var(--text-muted);font-size:0.82rem;">Could not load samples</span>';
    }
}


// ═══════════════ UI RENDERING: SELECTORS ═══════════════

function renderModelSelector() {
    const container = document.getElementById('model-selector');
    container.innerHTML = '';

    availableModels.forEach(model => {
        const btn = document.createElement('div');
        btn.className = `model-option ${model.key === activeModelKey ? 'active' : ''}`;
        
        let scoreStr = '';
        if (model.key !== 'ensemble') {
            scoreStr = `<div class="model-option-score">R²: ${model.r2.toFixed(3)} | MAE: ${model.mae.toFixed(1)}y</div>`;
        } else {
            scoreStr = `<div class="model-option-score" style="color:var(--green)">Best Overall Performance</div>`;
        }

        btn.innerHTML = `
            <div class="model-option-name">${model.name}</div>
            <div class="model-option-desc">${model.description}</div>
            ${scoreStr}
        `;
        
        btn.addEventListener('click', () => {
            activeModelKey = model.key;
            document.querySelectorAll('.model-option').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
        
        container.appendChild(btn);
    });
}

function renderSampleButtons() {
    const container = document.getElementById('sample-buttons');
    if (sampleData.length === 0) {
        container.innerHTML = '<span style="color:var(--text-muted);font-size:0.82rem;">No samples available</span>';
        return;
    }

    container.innerHTML = '';
    sampleData.forEach((s, i) => {
        const btn = document.createElement('button');
        btn.className = 'sample-btn';
        btn.type = 'button';
        btn.textContent = `${s.subject_id} (${s.gender}, ${Math.round(s.chronological_age)}y)`;
        btn.addEventListener('click', () => loadSampleData(i));
        container.appendChild(btn);
    });
}

function loadSampleData(index) {
    const sample = sampleData[index];
    if (!sample) return;

    // Fill age
    const ageInput = document.getElementById('chronological_age');
    ageInput.value = sample.chronological_age;
    ageInput.dispatchEvent(new Event('input', { bubbles: true }));

    // Fill features
    for (const [fname, val] of Object.entries(sample.features)) {
        const input = document.getElementById(fname);
        if (input) {
            input.value = val;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    // Visual feedback
    const btns = document.querySelectorAll('.sample-btn');
    btns.forEach((b, i) => {
        b.style.background = i === index ? 'rgba(99,102,241,0.3)' : '';
        b.style.borderColor = i === index ? 'var(--accent-primary)' : '';
    });
}


// ═══════════════ FORM HANDLING ═══════════════

function setupFormHandlers() {
    const form = document.getElementById('prediction-form');
    const clearBtn = document.getElementById('clear-btn');

    form.addEventListener('submit', handlePredict);
    clearBtn.addEventListener('click', clearForm);
}

function clearForm() {
    document.getElementById('prediction-form').reset();
    document.getElementById('results-section').style.display = 'none';
    document.querySelectorAll('.sample-btn').forEach(b => {
        b.style.background = '';
        b.style.borderColor = '';
    });
}

async function handlePredict(e) {
    e.preventDefault();

    const predictBtn = document.getElementById('predict-btn');
    const workbench = document.getElementById('analysis-workbench');
    
    // Validate
    const features = {};
    for (const fname of Object.keys(FEATURE_DEFAULTS)) {
        const input = document.getElementById(fname);
        if (!input.value) {
            input.focus();
            input.style.borderColor = 'var(--red)';
            setTimeout(() => input.style.borderColor = '', 2000);
            return;
        }
        features[fname] = parseFloat(input.value);
    }

    const ageInput = document.getElementById('chronological_age');
    const chronologicalAge = ageInput.value ? parseFloat(ageInput.value) : null;

    // Show Workbench
    workbench.style.display = 'flex';
    resetWorkbench();

    try {
        const res = await fetch('/api/predict_async', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                features, 
                chronological_age: chronologicalAge,
                model: activeModelKey
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Request failed');
        }

        const { task_id } = await res.json();
        pollTaskStatus(task_id);

    } catch (err) {
        workbench.style.display = 'none';
        alert('Error: ' + err.message);
    }
}

function resetWorkbench() {
    document.getElementById('workbench-progress').style.width = '0%';
    document.getElementById('workbench-status').textContent = 'Initializing neural engines...';
    document.querySelectorAll('.workbench-steps .step').forEach(s => {
        s.classList.remove('active', 'completed');
    });
}

async function pollTaskStatus(taskId) {
    try {
        const res = await fetch(`/api/task_status/${taskId}`);
        const status = await res.json();

        if (status.status === 'completed') {
            document.getElementById('workbench-progress').style.width = '100%';
            setTimeout(() => {
                document.getElementById('analysis-workbench').style.display = 'none';
                displayResults(status.result);
            }, 800);
            return;
        }

        if (status.status === 'failed') {
            document.getElementById('analysis-workbench').style.display = 'none';
            alert('Analysis Failed: ' + status.error);
            return;
        }

        // Update UI
        const progress = status.progress || 0;
        document.getElementById('workbench-progress').style.width = `${progress}%`;
        document.getElementById('workbench-status').textContent = status.message;

        // Step Highlighting
        const steps = document.querySelectorAll('.workbench-steps .step');
        if (progress > 10) steps[0].classList.add('active');
        if (progress > 30) { steps[0].classList.add('completed'); steps[1].classList.add('active'); }
        if (progress > 50) { steps[1].classList.add('completed'); steps[2].classList.add('active'); }
        if (progress > 85) { steps[2].classList.add('completed'); steps[3].classList.add('active'); }
        if (progress > 96) { steps[3].classList.add('completed'); }

        // Poll again
        setTimeout(() => pollTaskStatus(taskId), 800);

    } catch (e) {
        console.error("Polling error:", e);
        setTimeout(() => pollTaskStatus(taskId), 2000);
    }
}


// ═══════════════ DISPLAY RESULTS ═══════════════

function displayResults(result) {
    const section = document.getElementById('results-section');
    section.style.display = 'block';

    // Smooth scroll to results
    setTimeout(() => {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Gauge
    drawGauge(result.predicted_age);
    document.getElementById('gauge-value').textContent = result.predicted_age.toFixed(1);

    // Brain age gap
    displayGap(result);

    // Model info
    document.getElementById('active-model-name').textContent = result.model_name;
    document.getElementById('r2-score').textContent = result.model_r2.toFixed(4);
    document.getElementById('cv-r2-score').textContent = result.model_cv_r2.toFixed(4);
    document.getElementById('mae-score').textContent = result.model_mae.toFixed(2) + 'y';

    // Comparison Table
    renderComparisonTable(result);

    // Assessment
    displayAssessment(result.recommendations.overall_assessment);

    // Charts
    drawSHAPChart(result.feature_contributions, result.base_value, result.predicted_age, result.chronological_age);

    // Recommendations (RL prioritized)
    displayRecommendations(result);

    // Phase 3: Spatial Heatmap & Longitudinal Tracking
    updateSpatialHeatmap(result.feature_contributions);
    saveToPatientHistory(result);
    renderPatientHistory();

    // Phase 4: Full Clinical & Research Analytics
    if (result.risk_category) displayRiskCategory(result.risk_category);
    if (result.multi_target) displayMultiTarget(result.multi_target);
    displayConfidence(result.ensemble_uncertainty, result.mc_dropout, result.meta_model_selection);
    
    if (result.research) {
        if (result.research.causal) {
            const causal = result.research.causal;
            const ce = causal.causal_effect || 0;
            const causalEl = document.getElementById('causal-ate');
            causalEl.innerHTML = `
                <div style="font-size:1.6rem;font-weight:800;color:${ce <= 0 ? 'var(--green)' : 'var(--red)'}">${ce >= 0 ? '+' : ''}${ce.toFixed(2)} yrs</div>
                <div style="font-size:0.78rem;color:var(--text-muted);margin-top:0.5rem;line-height:1.5;">
                    <strong>Intervention:</strong> ${causal.intervention || 'do(Sleep = 8.5)'}<br>
                    <strong>Baseline Gap:</strong> ${(causal.baseline_gap || 0).toFixed(2)} yrs<br>
                    <strong>Post-Intervention Gap:</strong> ${(causal.average_outcome_gap || 0).toFixed(2)} yrs<br>
                    <strong>Interpretation:</strong> ${causal.interpretation || 'N/A'}
                </div>
            `;
        }
    }
    
    // Store result for 3D brain scene update
    window._lastResult = result;
    
    drawDiseaseRadar(result.disease_risk);
    drawTCAVChart(result.tcav_scores);
    drawTrajectoryProjection(result);
    displayCounterfactuals(result.counterfactuals);
    drawRegionRadar(result.temporal_segments);
    initTabNavigation();
}


function renderComparisonTable(result) {
    const tbody = document.getElementById('comparison-tbody');
    tbody.innerHTML = '';
    
    if (!result.all_model_scores) return;

    const scores = result.all_model_scores;
    const isEnsemble = result.model_key === 'ensemble';
    const activeKey = result.model_key;
    
    const modelKeys = Object.keys(scores);
    
    modelKeys.forEach(key => {
        const s = scores[key];
        const isActive = key === activeKey && !isEnsemble;
        
        let predVal = '--';
        if (isEnsemble && result.individual_predictions) {
            predVal = result.individual_predictions[key].toFixed(1) + ' yrs';
        } else if (key === activeKey) {
            predVal = result.predicted_age.toFixed(1) + ' yrs';
        }

        const tr = document.createElement('tr');
        if (isActive) tr.className = 'active-row';
        
        tr.innerHTML = `
            <td>${s.name} ${isActive ? '✓' : ''}</td>
            <td class="mono">${predVal}</td>
            <td class="mono">${s.r2.toFixed(4)}</td>
            <td class="mono">${s.cv_r2.toFixed(4)}</td>
            <td class="mono ${s.mae < 4.0 ? 'best-score' : ''}">${s.mae.toFixed(2)}y</td>
        `;
        tbody.appendChild(tr);
    });
    
    // Add Ensemble row if active
    if (isEnsemble) {
        const tr = document.createElement('tr');
        tr.className = 'active-row';
        tr.innerHTML = `
            <td><strong>${result.model_name} ✓</strong></td>
            <td class="mono"><strong>${result.predicted_age.toFixed(1)} yrs</strong></td>
            <td class="mono"><strong>${result.model_r2.toFixed(4)}</strong></td>
            <td class="mono"><strong>${result.model_cv_r2.toFixed(4)}</strong></td>
            <td class="mono best-score"><strong>${result.model_mae.toFixed(2)}y</strong></td>
        `;
        tbody.appendChild(tr);
    }
}


function displayGap(result) {
    const gapEl = document.getElementById('gap-value');
    const gapDesc = document.getElementById('gap-description');

    if (result.brain_age_gap !== null) {
        const gap = result.brain_age_gap;
        const sign = gap >= 0 ? '+' : '';
        gapEl.textContent = `${sign}${gap.toFixed(1)} yrs`;

        if (gap <= -2) {
            gapEl.style.color = 'var(--green)';
            gapDesc.textContent = `Your brain appears ${Math.abs(gap).toFixed(1)} years younger than your chronological age of ${result.chronological_age}.`;
        } else if (gap <= 2) {
            gapEl.style.color = 'var(--amber)';
            gapDesc.textContent = `Your brain age is close to your chronological age of ${result.chronological_age}. This is normal.`;
        } else {
            gapEl.style.color = 'var(--red)';
            gapDesc.textContent = `Your brain appears ${gap.toFixed(1)} years older than your chronological age of ${result.chronological_age}.`;
        }
    } else {
        gapEl.textContent = 'N/A';
        gapEl.style.color = 'var(--text-muted)';
        gapDesc.textContent = 'Enter your chronological age above to see the brain age gap.';
    }
}


function displayAssessment(assessment) {
    const card = document.getElementById('assessment-card');
    const icons = {
        excellent: '🌟', good: '✅', normal: '👍', attention: '⚠️', concern: '🔴', info: 'ℹ️'
    };

    document.getElementById('assessment-icon').textContent = icons[assessment.level] || '🧠';
    document.getElementById('assessment-title').textContent = assessment.title;
    document.getElementById('assessment-summary').textContent = assessment.summary;
    card.style.borderColor = assessment.color;
    card.style.boxShadow = `0 0 20px ${assessment.color}22`;
}


// ═══════════════ GAUGE DRAWING ═══════════════

function drawGauge(predictedAge) {
    const canvas = document.getElementById('gauge-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = 280 * dpr;
    canvas.height = 200 * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = '280px';
    canvas.style.height = '200px';

    const cx = 140, cy = 150;
    const radius = 110;
    const startAngle = Math.PI;
    const endAngle = 2 * Math.PI;

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(99,102,241,0.1)';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Age range: 20 to 85
    const minAge = 20, maxAge = 85;
    const clampedAge = Math.max(minAge, Math.min(maxAge, predictedAge));
    const fraction = (clampedAge - minAge) / (maxAge - minAge);
    const ageAngle = startAngle + fraction * Math.PI;

    // Colored arc
    const gradient = ctx.createLinearGradient(30, cy, 250, cy);
    gradient.addColorStop(0, '#22c55e');
    gradient.addColorStop(0.4, '#facc15');
    gradient.addColorStop(0.7, '#f97316');
    gradient.addColorStop(1, '#ef4444');

    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, ageAngle);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Needle dot
    const dotX = cx + radius * Math.cos(ageAngle);
    const dotY = cy + radius * Math.sin(ageAngle);
    ctx.beginPath();
    ctx.arc(dotX, dotY, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(dotX, dotY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = '#6366f1';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('20', 25, cy + 18);
    ctx.fillText('85', 255, cy + 18);
    ctx.fillText('50', cx, 30);
}


// ═══════════════ COMBINED SHAP & LIME CONTRIBUTIONS CHART ═══════════════

function drawSHAPChart(contributions, baseValue, predictedAge, chronologicalAge) {
    const canvas = document.getElementById('shap-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const w = canvas.parentElement.offsetWidth - 48;
    const h = Math.max(450, contributions.length * 45 + 100);

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    ctx.clearRect(0, 0, w, h);

    const isUsingAdjusted = chronologicalAge != null;
    const getShap = c => (isUsingAdjusted && c.adjusted_shap_value !== undefined) ? c.adjusted_shap_value : c.shap_value;
    const getLime = c => (isUsingAdjusted && c.adjusted_lime_value !== undefined) ? c.adjusted_lime_value : (c.lime_value || 0);

    // Sort by absolute SHAP value (ascending for bottom-to-top display, wait, I mean descending for top-to-bottom)
    const sorted = [...contributions].sort((a, b) => Math.abs(getShap(b)) - Math.abs(getShap(a)));

    const labelWidth = 200;
    const valueWidth = 60;
    const chartLeft = labelWidth + 10;
    const chartRight = w - valueWidth - 10;
    const chartWidth = chartRight - chartLeft;

    // Find extent of values for scaling based on BOTH shap and lime
    const allVals = [];
    sorted.forEach(c => {
        allVals.push(getShap(c));
        allVals.push(getLime(c));
    });
    
    const minVal = Math.min(...allVals, 0);
    const maxVal = Math.max(...allVals, 0);
    const extent = Math.max(Math.abs(minVal), Math.abs(maxVal)) * 1.3;

    const rowHeight = 36;
    const barHeight = 12;
    const barGap = 2;
    const topPad = 60;

    // Title & Legend Area
    ctx.textAlign = 'left';
    
    // Title
    ctx.fillStyle = '#94a3b8';
    ctx.font = '13px Inter, sans-serif';
    if (isUsingAdjusted) {
        ctx.fillText(`Age Baseline: ${chronologicalAge.toFixed(1)}y → Predicted: ${predictedAge.toFixed(1)}y`, chartLeft, 22);
    } else {
        ctx.fillText(`Population Baseline: ${baseValue.toFixed(1)}y → Predicted: ${predictedAge.toFixed(1)}y`, chartLeft, 22);
    }

    // Legend
    ctx.font = '11px Inter, sans-serif';
    
    // SHAP Aging
    ctx.fillStyle = 'rgba(239, 68, 68, 0.9)'; // Red
    ctx.beginPath(); ctx.roundRect(chartRight - 120, 10, 10, 10, 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0'; ctx.fillText('SHAP (Older)', chartRight - 105, 19);
    
    // SHAP Youthful
    ctx.fillStyle = 'rgba(34, 197, 94, 0.9)'; // Green
    ctx.beginPath(); ctx.roundRect(chartRight - 120, 26, 10, 10, 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0'; ctx.fillText('SHAP (Younger)', chartRight - 105, 35);
    
    // LIME Aging
    ctx.fillStyle = 'rgba(245, 158, 11, 0.9)'; // Orange
    ctx.beginPath(); ctx.roundRect(chartRight - 230, 10, 10, 10, 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0'; ctx.fillText('LIME (Older)', chartRight - 215, 19);
    
    // LIME Youthful
    ctx.fillStyle = 'rgba(14, 165, 233, 0.9)'; // Sky Blue
    ctx.beginPath(); ctx.roundRect(chartRight - 230, 26, 10, 10, 2); ctx.fill();
    ctx.fillStyle = '#e2e8f0'; ctx.fillText('LIME (Younger)', chartRight - 215, 35);

    // Zero line
    const zeroX = chartLeft + (chartWidth / 2);
    ctx.beginPath();
    ctx.moveTo(zeroX, topPad - 10);
    ctx.lineTo(zeroX, h - 30);
    ctx.strokeStyle = 'rgba(148,163,184,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw bars
    sorted.forEach((c, i) => {
        const rowY = topPad + i * rowHeight;
        
        const shapVal = getShap(c);
        const limeVal = getLime(c);
        
        const shapW = (Math.abs(shapVal) / extent) * (chartWidth / 2);
        const limeW = (Math.abs(limeVal) / extent) * (chartWidth / 2);

        // Label
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(c.display_name, labelWidth, rowY + rowHeight / 2 + 4);

        // ── SHAP Bar (Top) ──
        const isShapPositive = shapVal >= 0;
        const shapX = isShapPositive ? zeroX : zeroX - shapW;
        const shapY = rowY + 4;
        
        ctx.beginPath();
        ctx.roundRect(shapX, shapY, shapW, barHeight, 3);
        ctx.fillStyle = isShapPositive ? 'rgba(239, 68, 68, 0.85)' : 'rgba(34, 197, 94, 0.85)';
        ctx.fill();
        
        // SHAP Value Text
        ctx.fillStyle = isShapPositive ? '#fca5a5' : '#86efac';
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.textAlign = isShapPositive ? 'left' : 'right';
        ctx.fillText(
            `${isShapPositive ? '+' : ''}${shapVal.toFixed(2)}`,
            isShapPositive ? shapX + shapW + 6 : shapX - 6,
            shapY + barHeight - 2
        );

        // ── LIME Bar (Bottom) ──
        const isLimePositive = limeVal >= 0;
        const limeX = isLimePositive ? zeroX : zeroX - limeW;
        const limeY = shapY + barHeight + barGap;
        
        ctx.beginPath();
        ctx.roundRect(limeX, limeY, limeW, barHeight, 3);
        ctx.fillStyle = isLimePositive ? 'rgba(245, 158, 11, 0.85)' : 'rgba(14, 165, 233, 0.85)';
        ctx.fill();
        
        // LIME Value Text
        ctx.fillStyle = isLimePositive ? '#fcd34d' : '#7dd3fc';
        ctx.textAlign = isLimePositive ? 'left' : 'right';
        ctx.fillText(
            `${isLimePositive ? '+' : ''}${limeVal.toFixed(2)}`,
            isLimePositive ? limeX + limeW + 6 : limeX - 6,
            limeY + barHeight - 2
        );
    });

    // Bottom Axis Labels
    const legendY = h - 10;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#86efac';
    ctx.fillText('← Younger brain', chartLeft + chartWidth * 0.25, legendY);
    ctx.fillStyle = '#fca5a5';
    ctx.fillText('Older brain →', chartLeft + chartWidth * 0.75, legendY);
}



// ═══════════════ RECOMMENDATIONS DISPLAY ═══════════════

function displayRecommendations(result) {
    const recommendations = result.recommendations;
    const rlRecs = result.recommendations_rl || [];

    // RL Action Agent Section
    const rlContainer = document.getElementById('rl-recommendations');
    if (rlContainer) {
        rlContainer.innerHTML = '';
        if (rlRecs.length > 0) {
            rlRecs.forEach(rec => {
                const card = document.createElement('div');
                card.className = 'rl-rec-card';
                card.innerHTML = `
                    <div class="rl-impact">RL Action Agent Suggestion</div>
                    <div style="font-weight:700; color:#fff; margin-bottom:0.4rem;">${rec.action}</div>
                    <div style="font-size:0.85rem; color:var(--text-muted);">${rec.reasoning}</div>
                    <div style="font-size:0.75rem; color:var(--green); margin-top:0.4rem; font-family:monospace;">Predicted Reward: +${rec.reward.toFixed(2)} [Brain Age Reduction]</div>
                `;
                rlContainer.appendChild(card);
            });
        } else {
            rlContainer.innerHTML = '<div style="color:var(--text-muted); font-size:0.85rem;">RL Agent is gathering more baseline data...</div>';
        }
    }

    // Feature-specific Options
    const grid = document.getElementById('recommendations-grid');
    if (grid) {
        grid.innerHTML = '';
        const featureRecs = recommendations.feature_recommendations || [];
        const ordered = [
            ...featureRecs.filter(r => r.status === 'attention'),
            ...featureRecs.filter(r => r.status === 'positive'),
            ...featureRecs.filter(r => r.status === 'neutral')
        ];
        ordered.forEach(rec => {
            const card = document.createElement('div');
            card.className = `rec-card status-${rec.status}`;
            card.innerHTML = `
                <div class="rec-header">
                    <span class="rec-icon">${rec.icon}</span>
                    <span class="rec-region">${rec.region}</span>
                    <span class="rec-status">${rec.status_label}</span>
                </div>
                <div class="rec-value" style="line-height:1.4;">
                    Value: ${rec.value}${rec.unit ? ' ' + rec.unit : ''} · Raw Impact: SHAP ${rec.shap_value >= 0 ? '+' : ''}${(rec.shap_value||0).toFixed(2)}y<br>
                </div>
                <ul class="rec-tips">
                    ${rec.tips.map(t => `<li>${t}</li>`).join('')}
                </ul>
            `;
            grid.appendChild(card);
        });
    }

    // General tips
    const tipsContainer = document.getElementById('general-tips');
    tipsContainer.innerHTML = '';

    const generalTips = recommendations.general_tips || [];
    generalTips.forEach(tip => {
        const card = document.createElement('div');
        card.className = 'tip-card';
        card.innerHTML = `
            <span class="tip-icon">${tip.icon}</span>
            <div>
                <div class="tip-title">${tip.title}</div>
                <div class="tip-detail">${tip.detail}</div>
            </div>
        `;
        tipsContainer.appendChild(card);
    });
}


// ═══════════════ PHASE 3: SPATIAL HEATMAP & TRACKING ═══════════════

function updateSpatialHeatmap(contributions) {
    const regionScores = {
        "Frontal": 0,
        "Central": 0,
        "Temporal": 0,
        "Parietal": 0,
        "Occipital": 0
    };
    
    // Aggregate SHAP values per region
    contributions.forEach(c => {
        const region = c.feature.split('_')[0];
        if (regionScores[region] !== undefined) {
            regionScores[region] += c.shap_value;
        }
    });

    Object.keys(regionScores).forEach(region => {
        const path = document.getElementById(`region-${region}`);
        if (!path) return;
        
        const score = regionScores[region];
        
        // Dynamic Fill mapping
        // Negative (Protective) -> Green, Positive (Accelerating) -> Red, Neutral -> Dark Slate
        let color = '#2d3748';
        if (score > 1.5) color = '#7f1d1d';
        else if (score > 0.8) color = '#b91c1c';
        else if (score > 0.2) color = '#ef4444';
        else if (score < -1.5) color = '#14532d';
        else if (score < -0.8) color = '#15803d';
        else if (score < -0.2) color = '#22c55e';
        
        path.style.fill = color;
        
        // Add exact score to tooltip title
        const titleEl = path.querySelector("title");
        if (titleEl) {
            const sign = score > 0 ? "+" : "";
            titleEl.textContent = `${region} Lobe: ${sign}${score.toFixed(2)} yrs impact`;
        }
    });
}


let historyChartInstance = null;

function saveToPatientHistory(res) {
    let history = JSON.parse(localStorage.getItem('neuroage_history') || '[]');
    
    const entry = {
        date: new Date().toISOString(),
        chronological_age: res.chronological_age || null,
        predicted_age: res.predicted_age,
        gap: res.brain_age_gap || 0,
        model: res.model_name
    };
    
    history.push(entry);
    
    // Keep the latest 15 runs
    if (history.length > 15) history = history.slice(-15);
    
    localStorage.setItem('neuroage_history', JSON.stringify(history));
}

function renderPatientHistory() {
    const history = JSON.parse(localStorage.getItem('neuroage_history') || '[]');
    if (history.length === 0) return;
    
    // Calculate Trend
    let trendHtml = "";
    if (history.length >= 2) {
        const first = history[0].gap;
        const last = history[history.length-1].gap;
        const diff = last - first;
        let trend = "Stable";
        let color = "#94a3b8";
        if (diff > 1) { trend = "⚠️ Accelerating Aging Trend"; color = "#ef4444"; }
        else if (diff < -1) { trend = "✅ Improving / Decelerating Trend"; color = "#22c55e"; }
        trendHtml = `<div style="font-size:0.8rem; font-weight:600; color:${color}; margin-bottom:0.8rem;">Trend: ${trend}</div>`;
    }

    const canvas = document.getElementById('history-canvas');
    if (!canvas) return;
    
    // Inject trend before canvas parent if not there
    let trendContainer = document.getElementById('history-trend-indicator');
    if (!trendContainer) {
        trendContainer = document.createElement('div');
        trendContainer.id = 'history-trend-indicator';
        canvas.parentElement.prepend(trendContainer);
    }
    trendContainer.innerHTML = trendHtml;
    const ctx = canvas.getContext('2d');
    
    if (historyChartInstance) {
        historyChartInstance.destroy();
    }
    
    const labels = history.map(h => {
        const d = new Date(h.date);
        return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
    });
    
    const predictedAgeData = history.map(h => h.predicted_age);
    const chronoAgeData = history.map(h => h.chronological_age || h.predicted_age);
    
    historyChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted Brain Age',
                    data: predictedAgeData,
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: '#f97316'
                },
                {
                    label: 'Chronological Age',
                    data: chronoAgeData,
                    borderColor: '#6366f1',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'Age (years)', color: '#64748b' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', maxRotation: 45, minRotation: 45 }
                }
            }
        }
    });
}

function clearPatientHistory() {
    localStorage.removeItem('neuroage_history');
    if (historyChartInstance) {
        historyChartInstance.destroy();
    }
    renderPatientHistory();
}


// ═══════════════ DOCTOR DASHBOARD RENDERERS ═══════════════

function displayRiskCategory(risk) {
    if (!risk) return;
    const badge = document.getElementById('risk-badge');
    const severity = document.getElementById('risk-severity');
    const desc = document.getElementById('risk-description');
    
    badge.textContent = risk.category;
    badge.style.color = risk.color;
    severity.textContent = `Severity: ${risk.severity}`;
    severity.style.color = risk.color;
    desc.textContent = risk.description;
    
    // Animate card border
    const card = document.getElementById('risk-card');
    card.style.borderColor = risk.color;
    card.style.borderWidth = '2px';
    card.style.borderStyle = 'solid';
}

function displayMultiTarget(mt) {
    if (!mt) return;
    document.getElementById('cognitive-score-val').textContent = `${mt.cognitive_score}/100`;
    document.getElementById('cognitive-label-val').textContent = mt.cognitive_label;
    document.getElementById('risk-score-val').textContent = `${mt.risk_score}/10`;
    document.getElementById('risk-label-val').textContent = mt.risk_label;
    
    // Color code
    const cogEl = document.getElementById('cognitive-score-val');
    cogEl.style.color = mt.cognitive_score > 70 ? '#22c55e' : (mt.cognitive_score > 50 ? '#f59e0b' : '#ef4444');
    const riskEl = document.getElementById('risk-score-val');
    riskEl.style.color = mt.risk_score > 6 ? '#ef4444' : (mt.risk_score > 3 ? '#f59e0b' : '#22c55e');
}

function displayConfidence(ensemble, mc, meta) {
    if (ensemble) {
        document.getElementById('ensemble-agreement').textContent = ensemble.agreement;
        document.getElementById('ensemble-ci').textContent = `[${ensemble.ci_lower}, ${ensemble.ci_upper}]`;
        document.getElementById('confidence-pct').textContent = `${ensemble.confidence_percent}%`;
    }
    if (mc && mc.available) {
        document.getElementById('mc-ci').textContent = `[${mc.ci_lower}, ${mc.ci_upper}]`;
    } else {
        document.getElementById('mc-ci').textContent = 'N/A';
    }
    if (meta) {
        document.getElementById('meta-best-model').textContent = meta.recommended_name;
    }
}

let diseaseRadarInstance = null;
function drawDiseaseRadar(disease) {
    if (!disease) return;
    const canvas = document.getElementById('disease-radar-chart');
    if (!canvas) return;
    
    if (diseaseRadarInstance) diseaseRadarInstance.destroy();
    
    diseaseRadarInstance = new Chart(canvas.getContext('2d'), {
        type: 'radar',
        data: {
            labels: ["Alzheimer's", "MCI", "Cognitive Decline", "Depression"],
            datasets: [{
                label: 'Risk %',
                data: [
                    disease.alzheimers.risk_percent,
                    disease.mci.risk_percent,
                    disease.cognitive_decline.risk_percent,
                    disease.depression_marker.risk_percent
                ],
                backgroundColor: 'rgba(239, 68, 68, 0.15)',
                borderColor: '#ef4444',
                borderWidth: 2,
                pointBackgroundColor: ['#ef4444', '#f59e0b', '#f97316', '#8b5cf6']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { stepSize: 25, color: '#64748b', backdropColor: 'transparent' },
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    pointLabels: { color: '#94a3b8', font: { size: 11 } }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
    
    // Disease details text
    const details = document.getElementById('disease-details');
    details.innerHTML = `
        <div style="margin-top:0.5rem;">
            <span style="color:${disease.alzheimers.color}">● Alzheimer's: ${disease.alzheimers.risk_percent}% (${disease.alzheimers.level})</span><br>
            <span style="color:${disease.mci.color}">● MCI: ${disease.mci.risk_percent}% (${disease.mci.level})</span><br>
            <span style="color:${disease.cognitive_decline.color}">● Decline: ${disease.cognitive_decline.risk_percent}% (${disease.cognitive_decline.level})</span><br>
            <span style="color:${disease.depression_marker.color}">● Depression: ${disease.depression_marker.risk_percent}% (${disease.depression_marker.level})</span>
        </div>
        <div style="margin-top:0.4rem;font-size:0.65rem;opacity:0.7;">θ/α ratio: ${disease.theta_alpha_ratio}</div>
    `;
}

let tcavChartInstance = null;
function drawTCAVChart(tcav) {
    if (!tcav || tcav.length === 0) return;
    const canvas = document.getElementById('tcav-chart');
    if (!canvas) return;
    
    if (tcavChartInstance) tcavChartInstance.destroy();
    
    const labels = tcav.map(t => t.concept);
    const values = tcav.map(t => t.sensitivity);
    const colors = tcav.map(t => t.direction === 'aging' ? '#ef4444' : '#22c55e');
    
    tcavChartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sensitivity (years/unit)',
                data: values,
                backgroundColor: colors.map(c => c + '66'),
                borderColor: colors,
                borderWidth: 2
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'Sensitivity (years/unit)', color: '#64748b' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function displayCounterfactuals(cfs) {
    const list = document.getElementById('counterfactual-list');
    if (!list) return;
    list.innerHTML = '';
    
    if (!cfs || cfs.length === 0) {
        list.innerHTML = '<div style="color:var(--text-muted);font-size:0.82rem;">No actionable counterfactuals found.</div>';
        return;
    }
    
    cfs.forEach((cf, i) => {
        const card = document.createElement('div');
        card.style.cssText = 'padding:0.6rem;margin-bottom:0.5rem;background:rgba(99,102,241,0.08);border-radius:8px;border-left:3px solid #6366f1;';
        card.innerHTML = `
            <div style="font-size:0.78rem;font-weight:600;color:var(--text-primary);">${i+1}. ${cf.display_name}</div>
            <div style="font-size:0.72rem;color:var(--text-muted);margin-top:0.2rem;">
                ${cf.direction === 'decrease' ? '📉' : '📈'} ${cf.current_value} → ${cf.target_value} (${cf.change_percent > 0 ? '+' : ''}${cf.change_percent}%)
            </div>
            <div style="font-size:0.72rem;color:#22c55e;margin-top:0.15rem;">
                🎯 Reduces brain age by <strong>${cf.predicted_reduction} years</strong>
            </div>
        `;
        list.appendChild(card);
    });
}

let regionRadarInstance = null;
function drawRegionRadar(segments) {
    if (!segments || !segments.region_contributions) return;
    const canvas = document.getElementById('region-radar-chart');
    if (!canvas) return;
    
    if (regionRadarInstance) regionRadarInstance.destroy();
    
    const regions = Object.keys(segments.region_contributions);
    const impacts = regions.map(r => Math.abs(segments.region_contributions[r].total_impact));
    const colors = regions.map(r => segments.region_contributions[r].direction === 'aging' ? '#ef4444' : '#22c55e');
    
    regionRadarInstance = new Chart(canvas.getContext('2d'), {
        type: 'radar',
        data: {
            labels: regions,
            datasets: [{
                label: 'Absolute Impact (years)',
                data: impacts,
                backgroundColor: 'rgba(99, 102, 241, 0.15)',
                borderColor: '#6366f1',
                borderWidth: 2,
                pointBackgroundColor: colors,
                pointRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    ticks: { color: '#64748b', backdropColor: 'transparent' },
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    pointLabels: { color: '#94a3b8', font: { size: 12, weight: 'bold' } }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

let trajectoryInstance = null;
function drawTrajectoryProjection(result) {
    const canvas = document.getElementById('trajectory-chart');
    if (!canvas) return;
    if (trajectoryInstance) trajectoryInstance.destroy();
    
    const chronoAge = result.chronological_age || result.predicted_age;
    const predictedAge = result.predicted_age;
    const gap = predictedAge - chronoAge;
    
    // Project 10 years: chronological ages 
    const years = [];
    const chronoTrack = [];
    const brainTrack = [];
    const ciUpperTrack = [];
    const ciLowerTrack = [];
    const improvedTrack = [];
    
    const unc = result.ensemble_uncertainty || {};
    const std = unc.std || 1.5;
    
    // Check if backend prognosis data is available
    const prognosis = (result.research && result.research.prognosis) ? result.research.prognosis : null;
    
    for (let y = 0; y <= 10; y++) {
        years.push(`Year ${y}`);
        const cAge = chronoAge + y;
        chronoTrack.push(cAge);
        // Brain age projection: assume gap persists or slightly accelerates
        const accel = gap > 2 ? 0.15 : (gap < -2 ? -0.1 : 0);
        const projBrainAge = predictedAge + y * (1 + accel);
        brainTrack.push(Math.round(projBrainAge * 10) / 10);
        ciUpperTrack.push(Math.round((projBrainAge + 1.96 * std * (1 + y * 0.1)) * 10) / 10);
        ciLowerTrack.push(Math.round((projBrainAge - 1.96 * std * (1 + y * 0.1)) * 10) / 10);
        
        // Improved lifestyle trajectory from backend prognosis
        if (prognosis && prognosis[y - 1]) {
            improvedTrack.push(Math.round((chronoAge + prognosis[y - 1].projected_gap) * 10) / 10);
        } else if (y === 0) {
            improvedTrack.push(predictedAge);
        } else {
            // Fallback: simulate improved lifestyle (gap reduces by 0.3/year)
            const improvedGap = Math.max(gap - 0.3 * y, gap * 0.3);
            improvedTrack.push(Math.round((cAge + improvedGap) * 10) / 10);
        }
    }
    
    const datasets = [
        {
            label: 'Chronological Age',
            data: chronoTrack,
            borderColor: '#6366f1',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        },
        {
            label: 'Projected Brain Age (Current)',
            data: brainTrack,
            borderColor: '#f97316',
            backgroundColor: 'rgba(249, 115, 22, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.3,
            pointBackgroundColor: '#f97316'
        },
        {
            label: 'Improved Lifestyle Trajectory',
            data: improvedTrack,
            borderColor: '#22c55e',
            backgroundColor: 'rgba(34, 197, 94, 0.08)',
            borderWidth: 2,
            borderDash: [6, 3],
            fill: false,
            tension: 0.3,
            pointBackgroundColor: '#22c55e',
            pointRadius: 3
        },
        {
            label: '95% CI Upper',
            data: ciUpperTrack,
            borderColor: 'rgba(249, 115, 22, 0.3)',
            borderWidth: 1,
            borderDash: [3, 3],
            fill: '+1',
            backgroundColor: 'rgba(249, 115, 22, 0.06)',
            pointRadius: 0
        },
        {
            label: '95% CI Lower',
            data: ciLowerTrack,
            borderColor: 'rgba(249, 115, 22, 0.3)',
            borderWidth: 1,
            borderDash: [3, 3],
            fill: false,
            pointRadius: 0
        }
    ];
    
    trajectoryInstance = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: years, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'Age (years)', color: '#64748b' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

/** ═══════════════ RESEARCH SUITE RENDERING ═══════════════ **/

function initTabNavigation() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
            
            btn.classList.add('active');
            const target = btn.dataset.tab;
            document.getElementById(target + '-tab').style.display = 'block';

            if (target === 'research') {
                if (!threeRenderer) {
                    initThreeJS();
                } else if (window._lastResult && window._lastResult.feature_contributions) {
                    updateBrain3DColors(window._lastResult.feature_contributions);
                }
            }
        });
    });
}

function initThreeJS() {
    const container = document.getElementById('neuro-3d-container');
    if (!container) return;

    threeScene = new THREE.Scene();
    threeCamera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    threeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    threeRenderer.setPixelRatio(window.devicePixelRatio);
    threeRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(threeRenderer.domElement);

    // Geometry: Semi-transparent brain sphere
    const geometry = new THREE.IcosahedronGeometry(2.5, 3);
    const material = new THREE.MeshPhongMaterial({
        color: 0x6366f1,
        wireframe: true,
        transparent: true,
        opacity: 0.15,
        emissive: 0x2e1065
    });
    threeBrain = new THREE.Mesh(geometry, material);
    threeScene.add(threeBrain);

    // Region positions (approximate 3D brain anatomy)
    const regionPositions = {
        Frontal:   { x: 0,    y: 1.8,  z: 1.2 },
        Central:   { x: 0,    y: 2.2,  z: -0.2 },
        Temporal:  { x: 2.0,  y: 0,    z: 0.3 },
        Parietal:  { x: 0,    y: 1.5,  z: -1.5 },
        Occipital: { x: 0,    y: 0.5,  z: -2.2 }
    };

    // Create region spheres with labels
    window._regionSpheres = {};
    Object.entries(regionPositions).forEach(([region, pos]) => {
        const sphereGeo = new THREE.SphereGeometry(0.35, 16, 16);
        const sphereMat = new THREE.MeshPhongMaterial({
            color: 0x6366f1,
            transparent: true,
            opacity: 0.8,
            emissive: 0x1e1b4b
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.set(pos.x, pos.y, pos.z);
        threeBrain.add(sphere);
        window._regionSpheres[region] = sphere;

        // Add mirror for Temporal (bilateral)
        if (region === 'Temporal') {
            const mirrorSphere = new THREE.Mesh(sphereGeo.clone(), sphereMat.clone());
            mirrorSphere.position.set(-pos.x, pos.y, pos.z);
            threeBrain.add(mirrorSphere);
            window._regionSpheres['Temporal_R'] = mirrorSphere;
        }
    });

    // Apply SHAP-based coloring if result is available
    if (window._lastResult && window._lastResult.feature_contributions) {
        updateBrain3DColors(window._lastResult.feature_contributions);
    }

    // Lighting
    const light = new THREE.PointLight(0xffffff, 1.2, 100);
    light.position.set(5, 5, 5);
    threeScene.add(light);
    const light2 = new THREE.PointLight(0x8b5cf6, 0.5, 100);
    light2.position.set(-5, -3, 3);
    threeScene.add(light2);
    threeScene.add(new THREE.AmbientLight(0x404040, 0.8));

    threeCamera.position.z = 6;

    // Mouse rotation
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    container.addEventListener('mousedown', (e) => { isDragging = true; prevMouse = { x: e.clientX, y: e.clientY }; });
    container.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const dx = e.clientX - prevMouse.x;
        const dy = e.clientY - prevMouse.y;
        threeBrain.rotation.y += dx * 0.01;
        threeBrain.rotation.x += dy * 0.01;
        prevMouse = { x: e.clientX, y: e.clientY };
    });
    container.addEventListener('mouseup', () => { isDragging = false; });
    container.addEventListener('mouseleave', () => { isDragging = false; });

    function animate() {
        requestAnimationFrame(animate);
        if (!isDragging) {
            threeBrain.rotation.y += 0.003;
        }
        threeRenderer.render(threeScene, threeCamera);
    }
    animate();
}

function updateBrain3DColors(contributions) {
    if (!window._regionSpheres) return;
    
    const regionScores = { Frontal: 0, Central: 0, Temporal: 0, Parietal: 0, Occipital: 0 };
    contributions.forEach(c => {
        const region = c.feature.split('_')[0];
        if (regionScores[region] !== undefined) {
            regionScores[region] += c.shap_value;
        }
    });

    Object.entries(regionScores).forEach(([region, score]) => {
        const sphere = window._regionSpheres[region];
        if (!sphere) return;
        
        // Scale sphere size by absolute impact
        const scale = 0.35 + Math.min(0.5, Math.abs(score) * 0.15);
        sphere.scale.set(scale / 0.35, scale / 0.35, scale / 0.35);
        
        // Color: red for aging, green for protective
        let color, emissive;
        if (score > 1.0)      { color = 0xef4444; emissive = 0x7f1d1d; }
        else if (score > 0.3) { color = 0xf97316; emissive = 0x7c2d12; }
        else if (score < -1.0){ color = 0x22c55e; emissive = 0x14532d; }
        else if (score < -0.3){ color = 0x10b981; emissive = 0x064e3b; }
        else                  { color = 0x6366f1; emissive = 0x1e1b4b; }
        
        sphere.material.color.setHex(color);
        sphere.material.emissive.setHex(emissive);
        
        // Mirror for Temporal
        if (region === 'Temporal' && window._regionSpheres['Temporal_R']) {
            const mirror = window._regionSpheres['Temporal_R'];
            mirror.scale.copy(sphere.scale);
            mirror.material.color.setHex(color);
            mirror.material.emissive.setHex(emissive);
        }
    });
}


