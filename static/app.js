// Tab Switching Logic
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    event.currentTarget.classList.add('active');
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// Gallery Slide Switching Logic
function switchGallery(slideName) {
    document.querySelectorAll('.g-tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.gallery-slide').forEach(slide => slide.classList.remove('active'));
    
    event.currentTarget.classList.add('active');
    document.getElementById(`slide-${slideName}`).classList.add('active');
}

// Helper to reset and show results area
function prepareResultsPanel(mode) {
    document.getElementById('results-placeholder').classList.add('hidden');
    
    if (mode === 'single') {
        document.getElementById('results-content').classList.remove('hidden');
        document.getElementById('batch-results-content').classList.add('hidden');
    } else {
        document.getElementById('results-content').classList.add('hidden');
        document.getElementById('batch-results-content').classList.remove('hidden');
    }
}

// Render predictions for single record diagnostics
function renderSingleResults(res, metaText) {
    prepareResultsPanel('single');
    
    document.getElementById('meta-text').innerHTML = metaText;
    
    // 1. Random Forest Binary
    const rfBinVal = document.getElementById('rf-bin-val');
    rfBinVal.innerText = res.rf_binary.class;
    rfBinVal.className = 'val ' + (res.rf_binary.class === 'Normal' ? 'normal-state' : 'anomaly-state');
    
    const rfProbPct = Math.round(res.rf_binary.prob * 100);
    document.getElementById('rf-bin-prob').innerText = `${rfProbPct}%`;
    const rfFill = document.getElementById('rf-bin-fill');
    rfFill.style.width = `${rfProbPct}%`;
    rfFill.className = 'meter-fill' + (rfProbPct > 50 ? ' warning' : '');
    
    document.getElementById('rf-multi-val').innerText = res.rf_multi.class;
    
    // 2. SVM Binary
    const svmBinVal = document.getElementById('svm-bin-val');
    svmBinVal.innerText = res.svm_binary.class;
    svmBinVal.className = 'val ' + (res.svm_binary.class === 'Normal' ? 'normal-state' : 'anomaly-state');
    
    const svmProbPct = Math.round(res.svm_binary.prob * 100);
    document.getElementById('svm-bin-prob').innerText = `${svmProbPct}%`;
    const svmFill = document.getElementById('svm-bin-fill');
    svmFill.style.width = `${svmProbPct}%`;
    svmFill.className = 'meter-fill' + (svmProbPct > 50 ? ' warning' : '');
    
    document.getElementById('svm-multi-val').innerText = res.svm_multi.class;
}

// Fetch and predict record by index
async function inspectDatasetRecord() {
    const index = document.getElementById('record-index').value;
    if (index === '') {
        alert("Please enter a valid index.");
        return;
    }
    
    try {
        const response = await fetch(`/api/inspect?index=${index}`);
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        const meta = `<strong>Dataset Index:</strong> ${index} | ` + 
                     `<strong>Protocol:</strong> ${data.record.protocol_type} | ` + 
                     `<strong>Service:</strong> ${data.record.service} | ` +
                     `<strong>Flag:</strong> ${data.record.flag} <br>` +
                     `<strong>Actual Dataset Label:</strong> <span class="${data.actual_label === 'normal' ? 'text-normal' : 'text-anomaly'}">${data.actual_label}</span> ` +
                     `(Difficulty: ${data.actual_difficulty})`;
                     
        renderSingleResults(data.predictions, meta);
    } catch (e) {
        console.error(e);
        alert("Failed to analyze dataset record. Ensure server is running.");
    }
}

// Submit manual form data
async function submitManualDiagnostics() {
    const form = document.getElementById('manual-form');
    const formData = new FormData(form);
    const params = {};
    formData.forEach((value, key) => {
        // Convert to number if numeric keys
        if (['duration', 'src_bytes', 'dst_bytes', 'logged_in', 'count'].includes(key)) {
            params[key] = Number(value);
        } else {
            params[key] = value;
        }
    });
    
    try {
        const response = await fetch('/api/predict_manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const data = await response.json();
        
        const meta = `<strong>Simulation Test Connection:</strong> <br>` + 
                     `<strong>Protocol:</strong> ${params.protocol_type.toUpperCase()} | ` + 
                     `<strong>Service:</strong> ${params.service} | ` +
                     `<strong>Flag:</strong> ${params.flag} | ` +
                     `<strong>Bytes Sent:</strong> ${params.src_bytes} | ` +
                     `<strong>Logged In:</strong> ${params.logged_in === 1 ? 'Yes' : 'No'}`;
                     
        renderSingleResults(data, meta);
    } catch (e) {
        console.error(e);
        alert("Error sending diagnostics check.");
    }
}

// Upload CSV file and display predictions log
async function uploadCSV() {
    const fileInput = document.getElementById('csv-file');
    if (!fileInput.files.length) {
        alert("Please select a CSV file first.");
        return;
    }
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    try {
        const response = await fetch('/api/predict_csv', {
            method: 'POST',
            body: formData
        });
        const results = await response.json();
        
        if (results.error) {
            alert(results.error);
            return;
        }
        
        prepareResultsPanel('batch');
        
        const tbody = document.querySelector('#batch-table tbody');
        tbody.innerHTML = ''; // clear table
        
        results.forEach(row => {
            const tr = document.createElement('tr');
            
            const isAnomaly = row.Binary === 'Anomaly';
            
            tr.innerHTML = `
                <td><strong>${row.Row}</strong></td>
                <td><span class="${isAnomaly ? 'text-anomaly' : 'text-normal'}">${row.Binary}</span></td>
                <td style="text-transform: uppercase; font-weight: 600;">${row["Multi-class"]}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error(e);
        alert("Error processing CSV upload. Ensure correct format and server communication.");
    }
}
