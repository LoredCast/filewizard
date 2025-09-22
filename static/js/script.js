document.addEventListener('DOMContentLoaded', () => {
    // --- Constants ---
    const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB chunks

    // Allow server to provide API prefix (e.g. "/api/v1") via window.APP_CONFIG.api_base
    const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.api_base) ? window.APP_CONFIG.api_base.replace(/\/$/, '') : '';

    function apiUrl(path) {
        // path may start with or without a leading slash
        if (!path) return API_BASE || '/';
        if (path.startsWith('/')) {
            return `${API_BASE}${path}`;
        }
        return `${API_BASE}/${path}`;
    }

    // --- User Locale and Timezone Detection ---
    const USER_LOCALE = navigator.language || 'en-US';
    const USER_TIMEZONE = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const DATETIME_FORMAT_OPTIONS = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        timeZone: USER_TIMEZONE,
    };

    // --- Element Selectors ---
    const appContainer = document.getElementById('app-container');
    const loginContainer = document.getElementById('login-container');
    const loginButton = document.getElementById('login-button');

    // Main form elements
    const mainFileInput = document.getElementById('main-file-input');
    const mainFileName = document.getElementById('main-file-name');
    const mainOutputFormatSelect = document.getElementById('main-output-format-select');
    const mainModelSizeSelect = document.getElementById('main-model-size-select');
    const mainTtsModelSelect = document.getElementById('main-tts-model-select');
    const startConversionBtn = document.getElementById('start-conversion-btn');
    const startOcrBtn = document.getElementById('start-ocr-btn');
    const startTranscriptionBtn = document.getElementById('start-transcription-btn');
    const startTtsBtn = document.getElementById('start-tts-btn');

    const downloadSelectedBtn = document.getElementById('download-selected-btn');
    const selectAllJobsCheckbox = document.getElementById('select-all-jobs');
    const jobListBody = document.getElementById('job-list-body');

    // Drag and Drop Elements
    const dragOverlay = document.getElementById('drag-overlay');
    const actionDialog = document.getElementById('action-dialog');
    const dialogFileCount = document.getElementById('dialog-file-count');
    const dialogInitialView = document.getElementById('dialog-initial-actions');
    const dialogConvertView = document.getElementById('dialog-convert-view');
    const dialogTtsView = document.getElementById('dialog-tts-view');
    const dialogConvertBtn = document.getElementById('dialog-action-convert');
    const dialogOcrBtn = document.getElementById('dialog-action-ocr');
    const dialogTranscribeBtn = document.getElementById('dialog-action-transcribe');
    const dialogTtsBtn = document.getElementById('dialog-action-tts');
    const dialogCancelBtn = document.getElementById('dialog-action-cancel');
    const dialogStartConversionBtn = document.getElementById('dialog-start-conversion');
    const dialogStartTtsBtn = document.getElementById('dialog-start-tts');
    const dialogBackBtn = document.getElementById('dialog-back');
    const dialogBackTtsBtn = document.getElementById('dialog-back-tts');
    const dialogOutputFormatSelect = document.getElementById('dialog-output-format-select');
    const dialogTtsModelSelect = document.getElementById('dialog-tts-model-select');


    // --- State Variables ---
    let conversionChoices = null;
    let transcriptionChoices = null;
    let ttsChoices = null;
    let dialogConversionChoices = null;
    let dialogTtsChoices = null;
    let ttsModelsCache = []; // Cache for formatted TTS models list
    const activePolls = new Map();
    let stagedFiles = null;


    // --- Authentication-aware Fetch Wrapper ---
    async function authFetch(url, options = {}) {
        // Normalize URL through apiUrl() if a bare endpoint is provided
        if (typeof url === 'string' && url.startsWith('/')) {
            url = apiUrl(url);
        }

        // Add default options: include credentials and accept JSON by default
        options = Object.assign({}, options);
        if (!Object.prototype.hasOwnProperty.call(options, 'credentials')) {
            options.credentials = 'include';
        }
        options.headers = options.headers || {};
        if (!options.headers.Accept) options.headers.Accept = 'application/json';

        const response = await fetch(url, options);
        if (response.status === 401) {
            alert('Your session has expired. You will be redirected to the login page.');
            window.location.href = apiUrl('/login');
            throw new Error('Session expired');
        }
        return response;
    }


    // --- Helper Functions ---
    function formatBytes(bytes, decimals = 1) {
        if (!+bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    // --- Chunked Uploading Logic ---
    async function uploadFileInChunks(file, taskType, options = {}) {
        const uploadId = 'upload-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

        const preliminaryJob = {
            id: uploadId,
            status: 'uploading',
            progress: 0,
            original_filename: file.name,
            input_filesize: file.size,
            task_type: taskType,
            created_at: new Date().toISOString()
        };
        renderJobRow(preliminaryJob);

        for (let chunkNumber = 0; chunkNumber < totalChunks; chunkNumber++) {
            const start = chunkNumber * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, file.size);
            const chunk = file.slice(start, end);
            const formData = new FormData();
            formData.append('chunk', chunk, file.name);
            formData.append('upload_id', uploadId);
            formData.append('chunk_number', chunkNumber);

            try {
                const response = await authFetch('/upload/chunk', {
                    method: 'POST',
                    body: formData,
                });
                if (!response.ok) {
                    throw new Error(`Chunk upload failed with status: ${response.status}`);
                }
                const progress = Math.round(((chunkNumber + 1) / totalChunks) * 100);
                updateUploadProgress(uploadId, progress);
            } catch (error) {
                console.error(`Error uploading chunk ${chunkNumber} for ${file.name}:`, error);
                if (error.message !== 'Session expired') {
                    updateJobToFailedState(uploadId, `Upload failed: ${error.message}`);
                }
                return;
            }
        }

        try {
            const finalizePayload = {
                upload_id: uploadId,
                original_filename: file.name,
                total_chunks: totalChunks,
                task_type: taskType,
                ...options
            };
            const finalizeResponse = await authFetch('/upload/finalize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(finalizePayload),
            });
            if (!finalizeResponse.ok) {
                let errorData = {};
                try { errorData = await finalizeResponse.json(); } catch (e) {}
                throw new Error(errorData.detail || 'Finalization failed');
            }
            const result = await finalizeResponse.json();
            const uploadRow = document.getElementById(uploadId);
            if (uploadRow) {
                uploadRow.id = `job-${result.job_id}`;
                const statusCell = uploadRow.querySelector('td[data-label="Status"] .cell-value');
                if (statusCell) {
                    statusCell.innerHTML = `<span class="job-status-badge status-pending">Pending</span>`;
                }
            }
            startPolling(result.job_id);
        } catch (error) {
            console.error(`Error finalizing upload for ${file.name}:`, error);
            if (error.message !== 'Session expired') {
                updateJobToFailedState(uploadId, `Finalization failed: ${error.message}`);
            }
        }
    }

    function updateUploadProgress(uploadId, progress) {
        const row = document.getElementById(uploadId);
        if (row) {
            const progressBar = row.querySelector('.progress-bar');
            if (progressBar) {
                progressBar.style.width = `${progress}%`;
            }
        }
    }

    function updateJobToFailedState(jobId, errorMessage) {
        const row = document.getElementById(jobId);
        if (row) {
            const statusCell = row.querySelector('td[data-label="Status"] .cell-value');
            const actionCell = row.querySelector('td[data-label="Action"] .cell-value');
            if (statusCell) statusCell.innerHTML = `<span class="job-status-badge status-failed">Failed</span>`;
            if (actionCell) {
                const errorTitle = errorMessage ? ` title="${errorMessage.replace(/"/g, '&quot;')}"` : '';
                actionCell.innerHTML = `<span class="error-text"${errorTitle}>Failed</span>`;
            }
        }
    }

    // --- Centralized Task Request Handler ---
    async function handleTaskRequest(taskType) {
        if (mainFileInput.files.length === 0) {
            alert('Please choose one or more files first.');
            return;
        }

        const files = Array.from(mainFileInput.files);
        const options = {};

        if (taskType === 'conversion') {
            const selectedFormat = conversionChoices.getValue(true);
            if (!selectedFormat) {
                alert('Please select a format to convert to.');
                return;
            }
            options.output_format = selectedFormat;
        } else if (taskType === 'transcription') {
            const selectedModel = transcriptionChoices.getValue(true);
            options.model_size = selectedModel;
        } else if (taskType === 'tts') {
            const selectedModel = ttsChoices.getValue(true);
            if (!selectedModel) {
                alert('Please select a voice model.');
                return;
            }
            options.model_name = selectedModel;
        }


        // Disable buttons during upload process
        startConversionBtn.disabled = true;
        startOcrBtn.disabled = true;
        startTranscriptionBtn.disabled = true;
        startTtsBtn.disabled = true;

        const uploadPromises = files.map(file => uploadFileInChunks(file, taskType, options));
        await Promise.allSettled(uploadPromises);

        // Reset file input and re-enable buttons
        mainFileInput.value = ''; // Resets the file list
        updateFileName(mainFileInput, mainFileName);
        startConversionBtn.disabled = false;
        startOcrBtn.disabled = false;
        startTranscriptionBtn.disabled = false;
        startTtsBtn.disabled = false;
    }


    function setupDragAndDropListeners() {
        let dragCounter = 0;
        window.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dragCounter++;
            document.body.classList.add('dragging');
        });
        window.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter === 0) document.body.classList.remove('dragging');
        });
        window.addEventListener('dragover', (e) => e.preventDefault());
        window.addEventListener('drop', (e) => {
            e.preventDefault();
            dragCounter = 0;
            document.body.classList.remove('dragging');
            if (e.target === dragOverlay || dragOverlay.contains(e.target)) {
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    stagedFiles = e.dataTransfer.files;
                    showActionDialog();
                }
            }
        });
    }

    function showActionDialog() {
        dialogFileCount.textContent = stagedFiles.length;
        
        // Setup Conversion Dropdown
        dialogOutputFormatSelect.innerHTML = mainOutputFormatSelect.innerHTML;
        if (dialogConversionChoices) dialogConversionChoices.destroy();
        dialogConversionChoices = new Choices(dialogOutputFormatSelect, {
            searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a format...',
        });

        // Setup TTS Dropdown
        if (dialogTtsChoices) dialogTtsChoices.destroy();
        dialogTtsChoices = new Choices(dialogTtsModelSelect, {
            searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a voice...',
        });
        dialogTtsChoices.setChoices(ttsModelsCache, 'value', 'label', true);


        dialogInitialView.style.display = 'grid';
        dialogConvertView.style.display = 'none';
        dialogTtsView.style.display = 'none';
        actionDialog.classList.add('visible');
    }

    function closeActionDialog() {
        actionDialog.classList.remove('visible');
        stagedFiles = null;
        if (dialogConversionChoices) {
            dialogConversionChoices.destroy();
            dialogConversionChoices = null;
        }
        if (dialogTtsChoices) {
            dialogTtsChoices.destroy();
            dialogTtsChoices = null;
        }
    }
    
    // --- Dialog Button Listeners ---
    dialogConvertBtn.addEventListener('click', () => {
        dialogInitialView.style.display = 'none';
        dialogConvertView.style.display = 'block';
    });
    dialogTtsBtn.addEventListener('click', () => {
        dialogInitialView.style.display = 'none';
        dialogTtsView.style.display = 'block';
    });
    dialogBackBtn.addEventListener('click', () => {
        dialogInitialView.style.display = 'grid';
        dialogConvertView.style.display = 'none';
    });
    dialogBackTtsBtn.addEventListener('click', () => {
        dialogInitialView.style.display = 'grid';
        dialogTtsView.style.display = 'none';
    });
    dialogStartConversionBtn.addEventListener('click', () => handleDialogAction('conversion'));
    dialogStartTtsBtn.addEventListener('click', () => handleDialogAction('tts'));
    dialogOcrBtn.addEventListener('click', () => handleDialogAction('ocr'));
    dialogTranscribeBtn.addEventListener('click', () => handleDialogAction('transcription'));
    dialogCancelBtn.addEventListener('click', closeActionDialog);

    function handleDialogAction(action) {
        if (!stagedFiles) return;
        let options = {};
        if (action === 'conversion') {
            const selectedFormat = dialogConversionChoices.getValue(true);
            if (!selectedFormat) {
                alert('Please select a format to convert to.');
                return;
            }
            options.output_format = selectedFormat;
        } else if (action === 'transcription') {
            options.model_size = mainModelSizeSelect.value;
        } else if (action === 'tts') {
            const selectedModel = dialogTtsChoices.getValue(true);
            if (!selectedModel) {
                alert('Please select a voice model.');
                return;
            }
            options.model_name = selectedModel;
        }
        Array.from(stagedFiles).forEach(file => uploadFileInChunks(file, action, options));
        closeActionDialog();
    }

    // -----------------------
    // TTS models loader (robust)
    // -----------------------
    async function loadTtsModels() {
        try {
            const response = await authFetch('/api/v1/tts-voices');
            if (!response.ok) throw new Error('Failed to fetch TTS voices.');
            const voicesData = await response.json();

            // voicesData might be an object map { id: meta } or an array [{ id, name, language, ... }]
            const voicesArray = [];
            if (Array.isArray(voicesData)) {
                for (const v of voicesData) {
                    // Accept either { id, name, language } or { voice_id, title, locale }
                    const id = v.id || v.voice_id || v.voice || v.name || null;
                    const name = v.name || v.title || v.display_name || id || 'Unknown';
                    const lang = (v.language && (v.language.name_native || v.language.name)) || v.locale || (id ? id.split(/[_-]/)[0] : 'Unknown');
                    if (id) voicesArray.push({ id, name, lang });
                }
            } else if (voicesData && typeof voicesData === 'object') {
                for (const key in voicesData) {
                    if (!Object.prototype.hasOwnProperty.call(voicesData, key)) continue;
                    const v = voicesData[key];
                    const id = v.id || key;
                    const name = v.name || v.title || v.display_name || id;
                    const lang = (v.language && (v.language.name_native || v.language.name)) || v.locale || (id ? id.split(/[_-]/)[0] : 'Unknown');
                    voicesArray.push({ id, name, lang });
                }
            } else {
                throw new Error('Unexpected voices payload');
            }

            // Group by language
            const groups = {};
            for (const v of voicesArray) {
                const langLabel = v.lang || 'Unknown';
                if (!groups[langLabel]) {
                    groups[langLabel] = { label: langLabel, id: langLabel, disabled: false, choices: [] };
                }
                groups[langLabel].choices.push({
                    value: v.id,
                    label: `${v.name}`
                });
            }
            ttsModelsCache = Object.values(groups).sort((a,b) => a.label.localeCompare(b.label));
            // If ttsChoices exists, update it; otherwise the initializer will set choices
            if (ttsChoices) {
                ttsChoices.setChoices(ttsModelsCache, 'value', 'label', true);
            }
        } catch (error) {
            console.error("Couldn't load TTS voices:", error);
            if (error.message !== 'Session expired') {
                if (ttsChoices) {
                    ttsChoices.setChoices([{ value: '', label: 'Error loading voices', disabled: true }], 'value', 'label');
                }
            }
        }
    }
    
    function initializeSelectors() {
        if (conversionChoices) conversionChoices.destroy();
        conversionChoices = new Choices(mainOutputFormatSelect, {
            searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a format...',
        });
        const tools = window.APP_CONFIG.conversionTools || {};
        const choicesArray = [];
        for (const toolKey in tools) {
            const tool = tools[toolKey];
            const group = { label: tool.name, id: toolKey, disabled: false, choices: [] };
            for (const formatKey in tool.formats) {
                group.choices.push({ value: `${toolKey}_${formatKey}`, label: `${tool.name} - ${formatKey.toUpperCase()} (${tool.formats[formatKey]})` });
            }
            choicesArray.push(group);
        }
        conversionChoices.setChoices(choicesArray, 'value', 'label', true);

        if (transcriptionChoices) transcriptionChoices.destroy();
        transcriptionChoices = new Choices(mainModelSizeSelect, {
            searchEnabled: false, shouldSort: false, itemSelectText: '',
        });

        if (ttsChoices) ttsChoices.destroy();
        ttsChoices = new Choices(mainTtsModelSelect, {
             searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select voice...',
        });
        loadTtsModels();
    }

    function updateFileName(input, nameDisplay) {
        const numFiles = input.files.length;
        let displayText = numFiles === 1 ? input.files[0].name : `${numFiles} files selected`;
        let displayTitle = numFiles > 1 ? Array.from(input.files).map(f => f.name).join(', ') : displayText;
        if (numFiles === 0) {
            displayText = 'No file chosen';
            displayTitle = 'No file chosen';
        }
        nameDisplay.textContent = displayText;
        nameDisplay.title = displayTitle;
    }

    async function handleCancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) return;
        try {
            const response = await authFetch(`/job/${jobId}/cancel`, { method: 'POST' });
            if (!response.ok) {
                let errorData = {};
                try { errorData = await response.json(); } catch (e) {}
                throw new Error(errorData.detail || 'Failed to cancel job.');
            }
            stopPolling(jobId);
            const row = document.getElementById(`job-${jobId}`);
            if (row) {
                const statusCell = row.querySelector('td[data-label="Status"] .cell-value');
                const actionCell = row.querySelector('td[data-label="Action"] .cell-value');
                if (statusCell) statusCell.innerHTML = `<span class="job-status-badge status-cancelled">Cancelled</span>`;
                if (actionCell) actionCell.innerHTML = `<span>-</span>`;
            }
        } catch (error) {
            console.error('Error cancelling job:', error);
            if (error.message !== 'Session expired') alert(`Error: ${error.message}`);
        }
    }

    function handleSelectionChange() {
        const selectedCheckboxes = jobListBody.querySelectorAll('.job-checkbox:checked');
        downloadSelectedBtn.disabled = selectedCheckboxes.length === 0;

        const allCheckboxes = jobListBody.querySelectorAll('.job-checkbox');
        selectAllJobsCheckbox.checked = allCheckboxes.length > 0 && selectedCheckboxes.length === allCheckboxes.length;
    }

    async function handleBatchDownload() {
        const selectedIds = Array.from(jobListBody.querySelectorAll('.job-checkbox:checked')).map(cb => cb.value);
        if (selectedIds.length === 0) return;

        downloadSelectedBtn.disabled = true;
        downloadSelectedBtn.textContent = 'Zipping...';

        try {
            const response = await authFetch('/download/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_ids: selectedIds })
            });
            if (!response.ok) throw new Error('Batch download failed.');
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `file-wizard-batch-${Date.now()}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Batch download error:", error);
            alert("Could not download files. Please try again.");
        } finally {
            downloadSelectedBtn.disabled = false;
            downloadSelectedBtn.textContent = 'Download Selected as ZIP';
        }
    }

    async function loadInitialJobs() {
        try {
            const response = await authFetch('/jobs');
            if (!response.ok) throw new Error('Failed to fetch jobs.');
            let jobs = await response.json();
            
            // Sort jobs so parents come before children
            jobs.sort((a, b) => {
                if (a.id === b.parent_job_id) return -1;
                if (b.id === a.parent_job_id) return 1;
                return new Date(b.created_at) - new Date(a.created_at);
            });

            jobListBody.innerHTML = '';
            for (const job of jobs.reverse()) {
                renderJobRow(job);
                if (['pending', 'processing'].includes(job.status)) startPolling(job.id);
            }
            handleSelectionChange();
        } catch (error) {
            console.error("Couldn't load job history:", error);
            if (error.message !== 'Session expired') {
                jobListBody.innerHTML = '<tr><td colspan="7" style="text-align: center;">Could not load job history.</td></tr>';
            }
        }
    }
    
    // --- Polling and UI Rendering ---
    async function fetchAndRenderSubJobs(parentJobId) {
        try {
            const response = await authFetch('/jobs');
            if (!response.ok) return;
            const allJobs = await response.json();
            const childJobs = allJobs.filter(j => j.parent_job_id === parentJobId);

            for (const childJob of childJobs) {
                const childRowId = `job-${childJob.id}`;
                if (!document.getElementById(childRowId)) {
                    renderJobRow(childJob);
                    if (['pending', 'processing'].includes(childJob.status)) {
                        startPolling(childJob.id);
                    }
                }
            }
        } catch (error) {
            console.error(`Failed to fetch sub-jobs for parent ${parentJobId}:`, error);
        }
    }

    function startPolling(jobId) {
        if (activePolls.has(jobId)) return;

        const pollLogic = async () => {
            try {
                const response = await authFetch(`/job/${jobId}`);
                if (!response.ok) {
                    if (response.status === 404) stopPolling(jobId);
                    return;
                }
                const job = await response.json();
                renderJobRow(job);

                if (job.task_type === 'unzip' && job.status === 'processing') {
                    await fetchAndRenderSubJobs(job.id);
                }

                if (['completed', 'failed', 'cancelled'].includes(job.status)) {
                    stopPolling(jobId);
                }
            } catch (error) {
                console.error(`Error polling for job ${jobId}:`, error);
                stopPolling(jobId);
            }
        };

        const intervalId = setInterval(pollLogic, 3000);
        activePolls.set(jobId, intervalId);
        pollLogic(); // Run once immediately
    }


    function stopPolling(jobId) {
        if (activePolls.has(jobId)) {
            clearInterval(activePolls.get(jobId));
            activePolls.delete(jobId);
        }
    }

    function renderJobRow(job) {
        const rowId = job.id && String(job.id).startsWith('upload-') ? job.id : `job-${job.id}`;
        let row = document.getElementById(rowId);
        if (!row) {
            row = document.createElement('tr');
            row.id = rowId;
            const parentRow = job.parent_job_id ? document.getElementById(`job-${job.parent_job_id}`) : null;
            if (parentRow) {
                 parentRow.parentNode.insertBefore(row, parentRow.nextSibling);
            } else {
                jobListBody.prepend(row);
            }
        }

        let taskTypeLabel = job.task_type;
        if (job.task_type === 'conversion' && job.processed_filepath) {
            const extension = job.processed_filepath.split('.').pop();
            taskTypeLabel = `Convert to ${extension.toUpperCase()}`;
        } else if (job.task_type === 'tts') {
            taskTypeLabel = 'Synthesize Speech';
        } else if (job.task_type === 'unzip') {
            taskTypeLabel = 'Unpack ZIP';
        } else if (job.task_type) {
             taskTypeLabel = job.task_type.charAt(0).toUpperCase() + job.task_type.slice(1);
        }

        const submittedDate = job.created_at ? new Date(job.created_at) : new Date();
        const formattedDate = submittedDate.toLocaleString(USER_LOCALE, DATETIME_FORMAT_OPTIONS);

        let statusHtml = `<span class="job-status-badge status-${job.status}">${job.status}</span>`;
        if (job.status === 'uploading' || (job.status === 'processing' && job.task_type === 'unzip')) {
             statusHtml += `<div class="progress-bar-container"><div class="progress-bar" style="width: ${job.progress || 0}%"></div></div>`;
        } else if (job.status === 'processing') {
            const progressClass = (job.progress > 0) ? '' : 'indeterminate';
            const progressWidth = (job.progress > 0) ? job.progress : 100;
            statusHtml += `<div class="progress-bar-container"><div class="progress-bar ${progressClass}" style="width: ${progressWidth}%"></div></div>`;
        }

        let actionHtml = `<span>-</span>`;
        if (['pending', 'processing', 'uploading'].includes(job.status)) {
            actionHtml = `<button class="cancel-button" data-job-id="${job.id}">Cancel</button>`;
        } else if (job.status === 'completed') {
            if (job.task_type === 'unzip') {
                actionHtml = `<a href="${apiUrl('/download/zip-batch')}/${encodeURIComponent(job.id)}" class="download-button" download>Download Batch</a>`;
            } else if (job.processed_filepath) {
                const downloadFilename = job.processed_filepath.split(/[\\/]/).pop();
                actionHtml = `<a href="${apiUrl('/download')}/${encodeURIComponent(downloadFilename)}" class="download-button" download>Download</a>`;
            }
        } else if (job.status === 'failed') {
            const errorTitle = job.error_message ? ` title="${job.error_message.replace(/"/g, '&quot;')}"` : '';
            actionHtml = `<span class="error-text"${errorTitle}>Failed</span>`;
        }

        let fileSizeHtml = job.input_filesize ? formatBytes(job.input_filesize) : '-';
        if (job.status === 'completed' && job.output_filesize) {
            fileSizeHtml += ` → ${formatBytes(job.output_filesize)}`;
        }

        const escapedFilename = job.original_filename ? job.original_filename.replace(/</g, "&lt;").replace(/>/g, "&gt;") : "No filename";

        let checkboxHtml = '';
        if (job.status === 'completed' && job.processed_filepath && job.task_type !== 'unzip') {
            checkboxHtml = `<input type="checkbox" class="job-checkbox" value="${job.id}">`;
        }
        
        const rowClasses = [];
        if(job.parent_job_id) rowClasses.push('sub-job');
        if(job.task_type === 'unzip') rowClasses.push('parent-job');
        row.className = rowClasses.join(' ');
        if (job.parent_job_id) {
            row.dataset.parentId = job.parent_job_id;
        }

        const expanderHtml = job.task_type === 'unzip' ? '<span class="expander-arrow"></span>' : '';
        
        row.innerHTML = `
            <td data-label="Select"><span class="cell-value">${checkboxHtml}</span></td>
            <td data-label="File"><span class="cell-value" title="${escapedFilename}">${expanderHtml}<span class="file-cell-content">${escapedFilename}</span></span></td>
            <td data-label="File Size"><span class="cell-value">${fileSizeHtml}</span></td>
            <td data-label="Task"><span class="cell-value">${taskTypeLabel}</span></td>
            <td data-label="Submitted"><span class="cell-value">${formattedDate}</span></td>
            <td data-label="Status"><span class="cell-value status-cell-value">${statusHtml}</span></td>
            <td data-label="Action" class="action-col"><span class="cell-value">${actionHtml}</span></td>
        `;
    }

    // --- App Initialization and Auth Check ---
    function initializeApp() {
        if (appContainer) appContainer.style.display = 'block';
        if (loginContainer) loginContainer.style.display = 'none';

        startConversionBtn.addEventListener('click', () => handleTaskRequest('conversion'));
        startOcrBtn.addEventListener('click', () => handleTaskRequest('ocr'));
        startTranscriptionBtn.addEventListener('click', () => handleTaskRequest('transcription'));
        startTtsBtn.addEventListener('click', () => handleTaskRequest('tts'));
        mainFileInput.addEventListener('change', () => updateFileName(mainFileInput, mainFileName));

        jobListBody.addEventListener('click', (event) => {
            if (event.target.classList.contains('cancel-button')) {
                const jobId = event.target.dataset.jobId;
                handleCancelJob(jobId);
                return;
            }
             // Event delegation for collapsible rows
            const parentRow = event.target.closest('tr.parent-job');
            if (parentRow) {
                const parentId = parentRow.id.replace('job-', '');
                parentRow.classList.toggle('sub-jobs-visible');
                const areVisible = parentRow.classList.contains('sub-jobs-visible');

                // Toggle visibility of all child job rows
                const subJobs = jobListBody.querySelectorAll(`tr.sub-job[data-parent-id="${parentId}"]`);
                subJobs.forEach(subJob => {
                    // Use classes instead of direct style manipulation for robustness
                    if (areVisible) {
                        subJob.classList.add('is-visible');
                    } else {
                        subJob.classList.remove('is-visible');
                    }
                });
            }
        });

        jobListBody.addEventListener('change', (event) => {
            if (event.target.classList.contains('job-checkbox')) {
                handleSelectionChange();
            }
        });

        selectAllJobsCheckbox.addEventListener('change', () => {
            const isChecked = selectAllJobsCheckbox.checked;
            jobListBody.querySelectorAll('.job-checkbox').forEach(cb => {
                cb.checked = isChecked;
            });
            handleSelectionChange();
        });
        downloadSelectedBtn.addEventListener('click', handleBatchDownload);

        // Load initial data and setup UI components
        initializeSelectors();
        loadInitialJobs();
        setupDragAndDropListeners();
    }

    function showLoginView() {
        if (appContainer) appContainer.style.display = 'none';
        if (loginContainer) loginContainer.style.display = 'flex';
        if (loginButton) {
            loginButton.addEventListener('click', () => {
                window.location.href = apiUrl('/login');
            });
        }
    }

    // --- Entry Point ---
    if (window.APP_CONFIG && (window.APP_CONFIG.local_only_mode || window.APP_CONFIG.user)) {
        initializeApp();
    } else {
        showLoginView();
    }
});