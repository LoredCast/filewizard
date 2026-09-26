document.addEventListener('DOMContentLoaded', () => {
    // Server-side configuration is embedded as a JSON data block (see index.html).
    try {
        window.APP_CONFIG = JSON.parse(document.getElementById('app-config').textContent);
    } catch (e) {
        console.error('Could not read app configuration:', e);
        window.APP_CONFIG = {};
    }

    // --- Constants ---
    const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB chunks
    const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.api_base) ? window.APP_CONFIG.api_base.replace(/\/$/, '') : '';

    // --- User Locale ---
    // Browsers can report locale tags Intl rejects (e.g. "en-US@posix"), which would make every
    // toLocaleString() call throw and the job history fail to render; fall back to the default locale.
    const USER_LOCALE = (() => {
        try {
            const locale = navigator.language || 'en-US';
            new Intl.DateTimeFormat(locale);
            return locale;
        } catch (e) {
            return undefined;
        }
    })();
    const USER_TIMEZONE = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const DATETIME_FORMAT_OPTIONS = {
        year: 'numeric', month: 'short', day: 'numeric',
        hour: 'numeric', minute: '2-digit', timeZone: USER_TIMEZONE,
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
    const mainOcrLanguageSelect = document.getElementById('main-ocr-language-select');
    const startConversionBtn = document.getElementById('start-conversion-btn');
    const startOcrBtn = document.getElementById('start-ocr-btn');
    const startTranscriptionBtn = document.getElementById('start-transcription-btn');
    const startTtsBtn = document.getElementById('start-tts-btn');

    const downloadSelectedBtn = document.getElementById('download-selected-btn');
    const deleteSelectedBtn = document.getElementById('delete-selected-btn');
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
    let ocrLanguageChoices = null;
    let dialogConversionChoices = null;
    let dialogTtsChoices = null;
    let ttsModelsCache = [];
    let stagedFiles = null;
    let jobPollerInterval = null; // Polling timer
    const POLLING_INTERVAL_MS = 1500; // Check for updates every 1.5 seconds

    // --- Core Functions ---

    function apiUrl(path) {
        if (!path) return API_BASE || '/';
        return path.startsWith('/') ? `${API_BASE}${path}` : `${API_BASE}/${path}`;
    }

    async function authFetch(url, options = {}) {
        if (typeof url === 'string' && url.startsWith('/')) {
            url = apiUrl(url);
        }
        options = { credentials: 'include', ...options };
        options.headers = { Accept: 'application/json', ...options.headers };

        const response = await fetch(url, options);
        if (response.status === 401) {
            alert('Your session has expired. You will be redirected to the login page.');
            window.location.href = apiUrl('/login');
            throw new Error('Session expired');
        }
        return response;
    }

    function formatBytes(bytes, decimals = 1) {
        if (!+bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    async function pollForJobUpdates() {
        try {
            const allJobs = await authFetch('/jobs').then(res => res.json());

            const topLevelJobs = [];
            const childJobs = [];

            allJobs.forEach(job => {
                if (job.parent_job_id) {
                    childJobs.push(job);
                } else {
                    topLevelJobs.push(job);
                }
            });

            // Render top-level jobs first, then child jobs.
            // The renderJobRow function handles both creating and updating rows.
            topLevelJobs.forEach(job => renderJobRow(job));
            childJobs.forEach(job => renderJobRow(job));

            // Stop polling if there are no more active jobs.
            const hasActiveJobs = allJobs.some(job => ['pending', 'processing', 'uploading'].includes(job.status));
            if (!hasActiveJobs) {
                stopJobPolling();
            }
        } catch (error) {
            console.error("Job polling failed:", error);
            // Don't stop polling on error, just log it and retry next interval.
        }
    }

    function startJobPolling() {
        if (jobPollerInterval) return; // Poller is already running
        
        // Run once immediately, then start the regular interval
        pollForJobUpdates();
        jobPollerInterval = setInterval(pollForJobUpdates, POLLING_INTERVAL_MS);
    }

    function stopJobPolling() {
        if (jobPollerInterval) {
            clearInterval(jobPollerInterval);
            jobPollerInterval = null;
        }
    }

    // Escape a value for safe interpolation into HTML text or a quoted attribute.
    // Every server- or user-controlled string (file names, tool error output, ...) must go through this.
    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, ch => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[ch]));
    }

    function processedBasename(job) {
        return job.processed_filepath ? job.processed_filepath.split(/[\\\/]/).pop() : '';
    }

    function formatJobDate(job) {
        return new Date(job.created_at).toLocaleString(USER_LOCALE, DATETIME_FORMAT_OPTIONS);
    }

    function formatJobFileSize(job) {
        let text = job.input_filesize ? formatBytes(job.input_filesize) : '-';
        if (job.status === 'completed' && job.output_filesize) {
            text += ` → ${formatBytes(job.output_filesize)}`;
        }
        return text;
    }

    function buildDetailsHtml(job) {
        const processedName = processedBasename(job);
        const downloadHtml = (processedName && job.status === 'completed' && job.task_type !== 'unzip')
            ? `<div class="detail-item">
                    <span class="detail-label">Download:</span>
                    <a class="detail-value details-download-link" href="${escapeHtml(apiUrl('/download') + '/' + encodeURIComponent(processedName))}" download>${escapeHtml(processedName)}</a>
               </div>`
            : '';
        const errorHtml = job.error_message
            ? `<div class="detail-item">
                    <span class="detail-label">Error:</span>
                    <span class="detail-value error-text details-error" title="${escapeHtml(job.error_message)}">${escapeHtml(job.error_message.length > 50 ? job.error_message.substring(0, 50) + '...' : job.error_message)}</span>
               </div>`
            : '';
        return `
            <td colspan="7" class="job-details-content">
                <div class="job-details-grid">
                    <div class="detail-item">
                        <span class="detail-label">Full Filename:</span>
                        <span class="detail-value details-full-filename">${escapeHtml(job.original_filename || 'No filename')}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Submitted:</span>
                        <span class="detail-value details-submitted">${escapeHtml(formatJobDate(job))}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">File Size:</span>
                        <span class="detail-value details-file-size">${escapeHtml(formatJobFileSize(job))}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">ID:</span>
                        <span class="detail-value details-id">${escapeHtml(String(job.id).substring(0, 8))}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Processed File:</span>
                        <span class="detail-value">${escapeHtml(processedName || 'Not available')}</span>
                    </div>
                    ${downloadHtml}
                    ${errorHtml}
                </div>
            </td>
        `;
    }

    function renderJobRow(job) {
        const permanentDomId = `job-${job.id}`;
        let row = document.getElementById(permanentDomId);
        const jobId = escapeHtml(job.id);

        // --- Generate Content ---
        let taskTypeLabel = job.task_type;
        if (job.task_type === 'conversion' && job.processed_filepath) {
            const extension = job.processed_filepath.split('.').pop();
            taskTypeLabel = `Convert to ${extension.toUpperCase()}`;
        } else if (job.task_type === 'academic_pandoc') {
            taskTypeLabel = 'Academic PDF';
        } else if (job.task_type === 'tts') {
            taskTypeLabel = 'Synthesize Speech';
        } else if (job.task_type === 'unzip') {
            taskTypeLabel = 'Unpack ZIP';
        } else if (job.task_type) {
            taskTypeLabel = job.task_type.charAt(0).toUpperCase() + job.task_type.slice(1);
        }
        taskTypeLabel = escapeHtml(taskTypeLabel);
        const formattedDate = escapeHtml(formatJobDate(job));
        const status = escapeHtml(job.status);
        const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
        let statusHtml = `<span class="job-status-badge status-${status}">${status}</span>`;
        if ((job.status === 'processing' || job.status === 'pending') && job.task_type === 'unzip') {
            statusHtml += `<div class="progress-bar-container"><div class="progress-bar" style="width: ${progress}%"></div></div>`;
        } else if (job.status === 'processing') {
            const progressClass = (progress > 0) ? '' : 'indeterminate';
            const progressWidth = (progress > 0) ? progress : 100;
            statusHtml += `<div class="progress-bar-container"><div class="progress-bar ${progressClass}" style="width: ${progressWidth}%"></div></div>`;
        }
        let actionHtml = '<span>-</span>';
        if (['pending', 'processing', 'uploading'].includes(job.status)) {
            actionHtml = `<button class="cancel-button" data-job-id="${jobId}"><i class="fa">&#xf00d;</i></button>`;
        } else if (job.status === 'completed') {
            if (job.task_type === 'unzip') {
                actionHtml = `<a href="${escapeHtml(apiUrl('/download/zip-batch') + '/' + encodeURIComponent(job.id))}" class="download-button" download><i class="fa">&#xf019;</i> Batch</a>`;
            } else if (job.processed_filepath) {
                actionHtml = `<a href="${escapeHtml(apiUrl('/download') + '/' + encodeURIComponent(processedBasename(job)))}" class="download-button" download><i class="fa">&#xf019;</i></a>`;
            }
        } else if (job.status === 'failed') {
            const errorTitle = job.error_message ? ` title="${escapeHtml(job.error_message)}"` : '';
            actionHtml = `<span class="error-text"${errorTitle}>Error</span>`;
        } else if (job.status === 'cancelled') {
            actionHtml = `<span>Cancelled</span>`;
        }
        const fileSizeHtml = escapeHtml(formatJobFileSize(job));
        // Every finished job can be selected (for deletion); only jobs with a result file can be downloaded.
        let checkboxHtml = '';
        if (['completed', 'failed', 'cancelled'].includes(job.status)) {
            const downloadable = job.status === 'completed' && job.processed_filepath && job.task_type !== 'unzip';
            checkboxHtml = `<input type="checkbox" class="job-checkbox" value="${jobId}"${downloadable ? ' data-downloadable="1"' : ''}>`;
        }

        // Truncate filename for mobile view (truncate first, then escape, so entities are never cut in half)
        const rawFilename = job.original_filename || 'No filename';
        const escapedFilename = escapeHtml(rawFilename);
        const truncatedFilename = escapeHtml(rawFilename.length > 25 ? rawFilename.substring(0, 25) + '...' : rawFilename);
        const expanderHtml = job.task_type === 'unzip' ? '<span class="expander-arrow"></span>' : '';

        // --- Create or Update logic ---
        if (row) {
            // UPDATE an existing row
            const selectCell = row.querySelector('td[data-label="Select"] .cell-value');
            const fileCell = row.querySelector('td[data-label="File"] .cell-value');
            const taskCell = row.querySelector('td[data-label="Task"] .cell-value');
            const statusCell = row.querySelector('td[data-label="Status"] .cell-value');
            const actionCell = row.querySelector('td[data-label="Action"] .cell-value');

            if (selectCell) {
                // Keep the selection when a row is re-rendered by polling.
                const wasChecked = selectCell.querySelector('.job-checkbox')?.checked;
                selectCell.innerHTML = checkboxHtml;
                const checkbox = selectCell.querySelector('.job-checkbox');
                if (checkbox && wasChecked) checkbox.checked = true;
            }
            if (fileCell) {
                fileCell.innerHTML = `<span class="file-cell-content" title="${escapedFilename}">${expanderHtml}${truncatedFilename}</span><button class="details-button" style="display: none;" title="Show details">i</button>`;
            }
            if (taskCell) taskCell.innerHTML = taskTypeLabel;
            if (statusCell) statusCell.innerHTML = statusHtml;
            if (actionCell) actionCell.innerHTML = actionHtml;

            // Update the expanded details if they exist
            const detailsRow = document.getElementById(`${permanentDomId}-details`);
            if (detailsRow) {
                detailsRow.innerHTML = buildDetailsHtml(job);
            }
        } else {
            // CREATE a new row
            row = document.createElement('tr');
            row.id = permanentDomId;
            const rowClasses = [];
            if (job.parent_job_id) rowClasses.push('sub-job');
            if (job.task_type === 'unzip') rowClasses.push('parent-job');
            row.className = rowClasses.join(' ');
            if (job.parent_job_id) row.dataset.parentId = job.parent_job_id;

            // Create the row with all columns to match the table headers
            row.innerHTML = `
                <td data-label="Select"><span class="cell-value">${checkboxHtml}</span></td>
                <td data-label="File"><span class="cell-value" title="${escapedFilename}">${expanderHtml}<span class="file-cell-content">${truncatedFilename}</span><button class="details-button" style="display: none;" title="Show details">i</button></span></td>
                <td data-label="File Size"><span class="cell-value">${fileSizeHtml}</span></td>
                <td data-label="Task"><span class="cell-value">${taskTypeLabel}</span></td>
                <td data-label="Submitted"><span class="cell-value">${formattedDate}</span></td>
                <td data-label="Status"><span class="cell-value status-cell-value">${statusHtml}</span></td>
                <td data-label="Action" class="action-col"><span class="cell-value">${actionHtml}</span></td>
            `;
            const parentRow = job.parent_job_id ? document.getElementById(`job-${job.parent_job_id}`) : null;
            if (parentRow) {
                parentRow.parentNode.insertBefore(row, parentRow.nextSibling);
            } else {
                jobListBody.prepend(row);
            }

            // Create the details row (initially hidden)
            const detailsRow = document.createElement('tr');
            detailsRow.id = `${permanentDomId}-details`;
            detailsRow.className = 'job-details-row';
            detailsRow.style.display = 'none';
            detailsRow.innerHTML = buildDetailsHtml(job);

            // Insert details row after the main row
            row.parentNode.insertBefore(detailsRow, row.nextSibling);
        }
    }

    async function uploadFileInChunks(file, taskType, options = {}) {
        const uploadId = 'upload-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

        // Manually create and insert the temporary "uploading" row.
        const tempRow = document.createElement('tr');
        tempRow.id = uploadId;
        // Properly sanitize filename for XSS prevention
        const escapedFilename = file.name
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#x27;");
        const taskLabel = taskType.charAt(0).toUpperCase() + taskType.slice(1);
        tempRow.innerHTML = `
            <td data-label="Select"><span class="cell-value">-</span></td>
            <td data-label="File"><span class="cell-value" title="${escapedFilename}">${escapedFilename}</span></td>
            <td data-label="File Size"><span class="cell-value">${formatBytes(file.size)}</span></td>
            <td data-label="Task"><span class="cell-value">${taskLabel}</span></td>
            <td data-label="Submitted"><span class="cell-value">${new Date().toLocaleString(USER_LOCALE, DATETIME_FORMAT_OPTIONS)}</span></td>
            <td data-label="Status"><span class="cell-value status-cell-value">
                <span class="job-status-badge status-uploading">uploading</span>
                <div class="progress-bar-container"><div class="progress-bar" style="width: 0%"></div></div>
            </span></td>
            <td data-label="Action" class="action-col"><span class="cell-value">-</span></td>
        `;
        jobListBody.prepend(tempRow);

        // Upload chunks and update the progress bar directly.
        for (let chunkNumber = 0; chunkNumber < totalChunks; chunkNumber++) {
            const start = chunkNumber * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, file.size);
            const chunk = file.slice(start, end);
            const formData = new FormData();
            formData.append('chunk', chunk, file.name);
            formData.append('upload_id', uploadId);
            formData.append('chunk_number', chunkNumber);

            try {
                const response = await authFetch('/upload/chunk', { method: 'POST', body: formData });
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `Chunk upload failed (HTTP ${response.status})`);
                }
                const progress = Math.round(((chunkNumber + 1) / totalChunks) * 100);
                const progressBar = tempRow.querySelector('.progress-bar');
                if (progressBar) {  // Check if element still exists
                    progressBar.style.width = `${progress}%`;
                }
            } catch (error) {
                console.error(`Error uploading chunk ${chunkNumber}:`, error);
                const statusCell = tempRow.querySelector('.status-cell-value');
                if (statusCell) {  // Check if element still exists
                    statusCell.innerHTML = `<span class="job-status-badge status-failed" title="${escapeHtml(error.message)}">Upload Failed</span>`;
                }
                return; // Stop the upload process
            }
        }

        // Finalize the upload.
        try {
            const finalizePayload = { upload_id: uploadId, original_filename: file.name, total_chunks: totalChunks, task_type: taskType, ...options };
            const finalizeResponse = await authFetch('/upload/finalize', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(finalizePayload),
            });
            if (!finalizeResponse.ok) {
                const errorData = await finalizeResponse.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Finalization failed');
            }
            const result = await finalizeResponse.json();
            
            tempRow.remove();
            renderJobRow(result);
            startJobPolling();

        } catch (error) {
            console.error(`Error finalizing upload:`, error);
            const statusCell = tempRow.querySelector('.status-cell-value');
            if (statusCell) {  // Check if element still exists
                statusCell.innerHTML = `<span class="job-status-badge status-failed" title="${escapeHtml(error.message)}">Finalization Failed</span>`;
            }
        }
    }

    async function handleTaskRequest(taskType) {
        if (mainFileInput.files.length === 0) return alert('Please choose one or more files first.');
        const files = Array.from(mainFileInput.files);
        const options = {};

        if (taskType === 'conversion') {
            const selectedFormat = conversionChoices.getValue(true);
            if (!selectedFormat) return alert('Please select a format to convert to.');
            options.output_format = selectedFormat;
        } else if (taskType === 'transcription') {
            options.model_size = transcriptionChoices.getValue(true);
            options.generate_timestamps = document.getElementById('main-timestamps-checkbox').checked;
        } else if (taskType === 'tts') {
            const selectedModel = ttsChoices.getValue(true);
            if (!selectedModel) return alert('Please select a voice model.');
            options.model_name = selectedModel;
        } else if (taskType === 'ocr') {
            options.ocr_language = selectedOcrLanguage();
        }

        [startConversionBtn, startOcrBtn, startTranscriptionBtn, startTtsBtn].forEach(btn => btn.disabled = true);
        await Promise.allSettled(files.map(file => uploadFileInChunks(file, taskType, options)));
        mainFileInput.value = '';
        updateFileName(mainFileInput, mainFileName);
        [startConversionBtn, startOcrBtn, startTranscriptionBtn, startTtsBtn].forEach(btn => btn.disabled = false);
    }

    function setupDragAndDropListeners() {
        let dragCounter = 0;
        window.addEventListener('dragenter', e => { e.preventDefault(); dragCounter++; document.body.classList.add('dragging'); });
        window.addEventListener('dragleave', e => { e.preventDefault(); dragCounter--; if (dragCounter === 0) document.body.classList.remove('dragging'); });
        window.addEventListener('dragover', e => e.preventDefault());
        window.addEventListener('drop', e => {
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

    // Grouped choices for every configured output format (used when no single input type is known).
    function allFormatChoices() {
        const tools = window.APP_CONFIG.conversionTools || {};
        return Object.keys(tools).map(toolKey => {
            const tool = tools[toolKey];
            return {
                label: tool.name,
                choices: Object.keys(tool.formats || {}).map(formatKey => ({
                    value: `${toolKey}_${formatKey}`,
                    label: `${tool.name} - ${tool.formats[formatKey]}`
                }))
            };
        });
    }

    async function showActionDialog() {
        dialogFileCount.textContent = stagedFiles.length;

        // Create the widget first, then fill it: a single file only gets the formats its type supports.
        if (dialogConversionChoices) dialogConversionChoices.destroy();
        dialogConversionChoices = new Choices(dialogOutputFormatSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a format...' });
        if (stagedFiles.length === 1) {
            await updateFormatsForFile(stagedFiles[0], [dialogConversionChoices]);
        } else {
            dialogConversionChoices.setChoices(allFormatChoices(), 'value', 'label', true);
        }

        if (dialogTtsChoices) dialogTtsChoices.destroy();
        dialogTtsChoices = new Choices(dialogTtsModelSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a voice...' });
        dialogTtsChoices.setChoices(ttsModelsCache, 'value', 'label', true);
        dialogInitialView.style.display = 'grid';
        dialogConvertView.style.display = 'none';
        dialogTtsView.style.display = 'none';
        actionDialog.classList.add('visible');
    }

    function closeActionDialog() {
        actionDialog.classList.remove('visible');
        stagedFiles = null;
        if (dialogConversionChoices) { dialogConversionChoices.destroy(); dialogConversionChoices = null; }
        if (dialogTtsChoices) { dialogTtsChoices.destroy(); dialogTtsChoices = null; }
    }

    function handleDialogAction(action) {
        if (!stagedFiles) return;
        let options = {};
        if (action === 'conversion') {
            const selectedFormat = dialogConversionChoices.getValue(true);
            if (!selectedFormat) return alert('Please select a format to convert to.');
            options.output_format = selectedFormat;
        } else if (action === 'transcription') {
            options.model_size = mainModelSizeSelect.value;
            options.generate_timestamps = document.getElementById('dialog-timestamps-checkbox').checked;
        } else if (action === 'tts') {
            const selectedModel = dialogTtsChoices.getValue(true);
            if (!selectedModel) return alert('Please select a voice model.');
            options.model_name = selectedModel;
        } else if (action === 'ocr') {
            options.ocr_language = selectedOcrLanguage();
        }
        Array.from(stagedFiles).forEach(file => uploadFileInChunks(file, action, options));
        closeActionDialog();
    }

    // "eng+spa" style Tesseract spec from the language picker ("" = server default).
    function selectedOcrLanguage() {
        const values = ocrLanguageChoices ? ocrLanguageChoices.getValue(true) : [];
        return (Array.isArray(values) ? values : [values]).filter(Boolean).join('+');
    }

    async function loadOcrLanguages() {
        try {
            const data = await authFetch('/api/v1/ocr-languages').then(res => res.json());
            const defaults = String(data.default || 'eng').split('+');
            const choices = (data.languages || []).map(lang => ({ value: lang, label: lang, selected: defaults.includes(lang) }));
            if (ocrLanguageChoices) ocrLanguageChoices.setChoices(choices, 'value', 'label', true);
        } catch (error) {
            console.error("Couldn't load OCR languages:", error);
        }
    }

    async function loadTtsModels() {
        try {
            const voicesData = await authFetch('/api/v1/tts-voices').then(res => res.json());
            const voicesArray = [];
            if (Array.isArray(voicesData)) {
                voicesData.forEach(v => {
                    const id = v.id || v.voice_id || v.name;
                    if (id) voicesArray.push({ id, name: v.name || id, lang: (v.language && v.language.name) || v.locale || id.split(/[_-]/)[0] });
                });
            } else if (voicesData && typeof voicesData === 'object') {
                Object.keys(voicesData).forEach(key => {
                    const v = voicesData[key];
                    const id = v.id || key;
                    voicesArray.push({ id, name: v.name || id, lang: (v.language && v.language.name) || v.locale || id.split(/[_-]/)[0] });
                });
            }
            const groups = voicesArray.reduce((acc, v) => {
                const langLabel = v.lang || 'Unknown';
                if (!acc[langLabel]) acc[langLabel] = { label: langLabel, choices: [] };
                acc[langLabel].choices.push({ value: v.id, label: v.name });
                return acc;
            }, {});
            ttsModelsCache = Object.values(groups).sort((a, b) => a.label.localeCompare(b.label));
            if (ttsChoices) ttsChoices.setChoices(ttsModelsCache, 'value', 'label', true);
        } catch (error) {
            console.error("Couldn't load TTS voices:", error);
            if (ttsChoices && error.message !== 'Session expired') ttsChoices.setChoices([{ value: '', label: 'Error loading voices', disabled: true }], 'value', 'label');
        }
    }

function initializeSelectors() {
    if (conversionChoices) conversionChoices.destroy();
    conversionChoices = new Choices(mainOutputFormatSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a format...' });
    conversionChoices.setChoices(allFormatChoices(), 'value', 'label', true);

    if (transcriptionChoices) transcriptionChoices.destroy();
    transcriptionChoices = new Choices(mainModelSizeSelect, { searchEnabled: false, shouldSort: false, itemSelectText: '' });

    if (ttsChoices) ttsChoices.destroy();
    ttsChoices = new Choices(mainTtsModelSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select voice...' });
    loadTtsModels();

    if (!ocrLanguageChoices && mainOcrLanguageSelect) {
        ocrLanguageChoices = new Choices(mainOcrLanguageSelect, { removeItemButton: true, searchEnabled: true, itemSelectText: '', shouldSort: true, placeholder: true, placeholderValue: 'Server default' });
        loadOcrLanguages();
    }
}

    function getFileExtension(filename) {
        return '.' + filename.split('.').pop().toLowerCase();
    }

    // Restricts the given Choices widgets to the output formats available for the file's extension.
    async function updateFormatsForFile(file, targets) {
        if (!file) return;

        const fileExtension = getFileExtension(file.name);
        if (fileExtension === '.zip') {
            // ZIP uploads are processed file by file, so every output format applies.
            for (const choices of targets) {
                if (!choices) continue;
                choices.clearStore();
                choices.setChoices(allFormatChoices(), 'value', 'label', true);
            }
            return;
        }
        try {
            const response = await authFetch(`/api/v1/supported-formats/${encodeURIComponent(fileExtension)}`);
            if (!response.ok) {
                console.error(`Failed to fetch supported formats for ${fileExtension}:`, response.status);
                return;
            }

            const data = await response.json();
            const formats = data.formats || [];

            // Group formats by tool name for better UI
            const groupedFormats = formats.reduce((acc, format) => {
                if (!acc[format.tool]) {
                    acc[format.tool] = {
                        label: window.APP_CONFIG.conversionTools[format.tool]?.name || format.tool,
                        choices: []
                    };
                }
                acc[format.tool].choices.push({ value: format.value, label: format.label });
                return acc;
            }, {});
            const choicesArray = Object.values(groupedFormats);

            for (const choices of targets) {
                if (!choices) continue;
                choices.clearStore();
                choices.setChoices(choicesArray, 'value', 'label', true);
            }
        } catch (error) {
            console.error(`Error fetching supported formats for ${fileExtension}:`, error);
        }
    }

    function updateFileName(input, nameDisplay) {
        const numFiles = input.files.length;
        nameDisplay.textContent = numFiles === 1 ? input.files[0].name : (numFiles > 1 ? `${numFiles} files selected` : 'No files chosen');
        nameDisplay.title = numFiles > 1 ? Array.from(input.files).map(f => f.name).join(', ') : nameDisplay.textContent;
        
        // Update format dropdowns if exactly one file is selected
        if (numFiles === 1) {
            updateFormatsForFile(input.files[0], [conversionChoices]);
        } else if (numFiles > 1 && conversionChoices) {
            conversionChoices.clearStore();
            conversionChoices.setChoices(allFormatChoices(), 'value', 'label', true);
        } else if (numFiles === 0) {
            // Reset to all formats when no file is selected
            initializeSelectors();
        }
    }

    async function updateFormatCounts() {
        try {
            const response = await authFetch('/api/formats/count');
            if (!response.ok) {
                console.error('Failed to fetch format counts');
                return;
            }
            const data = await response.json();
            const inputCountEl = document.getElementById('input-format-count');
            const outputCountEl = document.getElementById('output-format-count');
            const counterEl = document.getElementById('format-counter');

            if (inputCountEl && outputCountEl && counterEl) {
                inputCountEl.textContent = data.input_formats_count;
                outputCountEl.textContent = data.output_formats_count;
                counterEl.style.display = 'flex';
            }
        } catch (error) {
            console.error('Error fetching format counts:', error);
        }
    }

    async function handleCancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) return;
        try {
            const response = await authFetch(`/job/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to cancel job.');
            }
            // Trigger a poll soon to see the "cancelled" status updated in the UI.
            setTimeout(pollForJobUpdates, 500);
        } catch (error) {
            if (error.message !== 'Session expired') alert(`Error: ${error.message}`);
        }
    }

    function handleSelectionChange() {
        const selectedCheckboxes = jobListBody.querySelectorAll('.job-checkbox:checked');
        downloadSelectedBtn.disabled = jobListBody.querySelectorAll('.job-checkbox[data-downloadable]:checked').length === 0;
        deleteSelectedBtn.disabled = selectedCheckboxes.length === 0;
        selectAllJobsCheckbox.checked = jobListBody.querySelectorAll('.job-checkbox').length > 0 && selectedCheckboxes.length === jobListBody.querySelectorAll('.job-checkbox').length;
    }

    function removeJobRows(jobId) {
        document.getElementById(`job-${jobId}`)?.remove();
        document.getElementById(`job-${jobId}-details`)?.remove();
    }

    async function handleBatchDelete() {
        const selectedIds = Array.from(jobListBody.querySelectorAll('.job-checkbox:checked')).map(cb => cb.value);
        if (selectedIds.length === 0) return;
        if (!confirm(`Delete ${selectedIds.length} job(s) and their files? ZIP batches are deleted with all their files.`)) return;
        deleteSelectedBtn.disabled = true;
        try {
            const response = await authFetch('/jobs/delete', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_ids: selectedIds })
            });
            if (!response.ok) throw new Error('Delete failed.');
            const result = await response.json();
            (result.deleted || []).forEach(removeJobRows);
        } catch (error) {
            console.error("Batch delete error:", error);
            if (error.message !== 'Session expired') alert("Could not delete the selected jobs. Please try again.");
        } finally {
            selectAllJobsCheckbox.checked = false;
            handleSelectionChange();
        }
    }

    async function handleBatchDownload() {
        const selectedIds = Array.from(jobListBody.querySelectorAll('.job-checkbox[data-downloadable]:checked')).map(cb => cb.value);
        if (selectedIds.length === 0) return;
        downloadSelectedBtn.disabled = true;
        downloadSelectedBtn.textContent = 'Zipping...';
        try {
            const response = await authFetch('/download/batch', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_ids: selectedIds })
            });
            if (!response.ok) throw new Error('Batch download failed.');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `file-wizard-batch-${Date.now()}.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
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
            const jobs = await response.json();
            jobListBody.innerHTML = '';
            jobs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at)); // Sort descending
            jobs.reverse().forEach(renderJobRow);
            handleSelectionChange();
            startJobPolling();
        } catch (error) {
            console.error("Couldn't load job history:", error);
            if (error.message !== 'Session expired') jobListBody.innerHTML = '<tr><td colspan="7" style="text-align: center;">Could not load job history.</td></tr>';
        }
    }

    function initializeApp() {
        // Check if required elements exist
        const requiredElements = [
            'app-container', 'main-file-input', 'job-list-body', 
            'start-conversion-btn', 'start-ocr-btn', 'start-transcription-btn', 'start-tts-btn'
        ];
        
        for (const id of requiredElements) {
            const element = document.getElementById(id);
            if (!element) {
                console.error(`Required element with ID "${id}" not found`);
                return;
            }
        }
        
        if (appContainer) appContainer.style.display = 'block';
        if (loginContainer) loginContainer.style.display = 'none';

        // Setup event listeners
        startConversionBtn.addEventListener('click', () => handleTaskRequest('conversion'));
        startOcrBtn.addEventListener('click', () => handleTaskRequest('ocr'));
        startTranscriptionBtn.addEventListener('click', () => handleTaskRequest('transcription'));
        startTtsBtn.addEventListener('click', () => handleTaskRequest('tts'));
        mainFileInput.addEventListener('change', () => updateFileName(mainFileInput, mainFileName));
        downloadSelectedBtn.addEventListener('click', handleBatchDownload);
        deleteSelectedBtn.addEventListener('click', handleBatchDelete);
        selectAllJobsCheckbox.addEventListener('change', () => {
            const checkboxes = jobListBody.querySelectorAll('.job-checkbox');
            checkboxes.forEach(checkbox => {
                checkbox.checked = selectAllJobsCheckbox.checked;
            });
            handleSelectionChange();
        });
        jobListBody.addEventListener('change', e => e.target.classList.contains('job-checkbox') && handleSelectionChange());
        jobListBody.addEventListener('click', e => {
            if (e.target.classList.contains('cancel-button')) {
                e.preventDefault();
                handleCancelJob(e.target.dataset.jobId);
            }
            const parentRow = e.target.closest('tr.parent-job');
            if (parentRow && !e.target.classList.contains('cancel-button') && !e.target.classList.contains('download-button')) {
                parentRow.classList.toggle('sub-jobs-visible');
                const areVisible = parentRow.classList.contains('sub-jobs-visible');
                jobListBody.querySelectorAll(`tr.sub-job[data-parent-id="${parentRow.id.replace('job-', '')}"]`)
                    .forEach(subJob => {
                        subJob.style.display = areVisible ? 'table-row' : 'none';
                    });
            }
            
            // Handle details button click
            if (e.target.classList.contains('details-button')) {
                const row = e.target.closest('tr');
                const jobId = row.id.replace('job-', '');
                const jobElement = row; // Get the job element to access its data
                let detailsRow = document.getElementById(`job-${jobId}-details`);
                
                // If details row doesn't exist, create it
                if (!detailsRow) {
                    // We need to get the job data - find it from the row's stored data or re-fetch
                    // For now, we'll just create a placeholder and the row will be updated when job data is refreshed
                    detailsRow = document.createElement('tr');
                    detailsRow.id = `job-${jobId}-details`;
                    detailsRow.className = 'job-details-row';
                    detailsRow.style.display = 'none';
                    
                    detailsRow.innerHTML = `
                        <td colspan="7" class="job-details-content">
                            <div class="job-details-grid">
                                <div class="detail-item">
                                    <span class="detail-label">Details loading...</span>
                                </div>
                            </div>
                        </td>
                    `;
                    
                    // Insert details row after the main row
                    row.parentNode.insertBefore(detailsRow, row.nextSibling);
                    
                    authFetch(`/job/${encodeURIComponent(jobId)}`)
                        .then(response => response.json())
                        .then(job => {
                            detailsRow.innerHTML = buildDetailsHtml(job);
                        })
                        .catch(error => {
                            console.error("Error fetching job details:", error);
                            detailsRow.innerHTML = `
                                <td colspan="7" class="job-details-content">
                                    <div class="job-details-grid">
                                        <div class="detail-item">
                                            <span class="detail-label">Error loading details</span>
                                        </div>
                                    </div>
                                </td>
                            `;
                        });
                }
                
                if (detailsRow) {
                    const isCurrentlyVisible = detailsRow.style.display !== 'none';
                    detailsRow.style.display = isCurrentlyVisible ? 'none' : 'table-row';
                    e.target.textContent = isCurrentlyVisible ? 'i' : '×';
                }
            }
        });

        // Dialog listeners
        dialogConvertBtn.addEventListener('click', () => { dialogInitialView.style.display = 'none'; dialogConvertView.style.display = 'block'; });
        dialogTtsBtn.addEventListener('click', () => { dialogInitialView.style.display = 'none'; dialogTtsView.style.display = 'block'; });
        dialogBackBtn.addEventListener('click', () => { dialogInitialView.style.display = 'grid'; dialogConvertView.style.display = 'none'; });
        dialogBackTtsBtn.addEventListener('click', () => { dialogInitialView.style.display = 'grid'; dialogTtsView.style.display = 'none'; });
        dialogStartConversionBtn.addEventListener('click', () => handleDialogAction('conversion'));
        dialogStartTtsBtn.addEventListener('click', () => handleDialogAction('tts'));
        dialogOcrBtn.addEventListener('click', () => handleDialogAction('ocr'));
        dialogTranscribeBtn.addEventListener('click', () => handleDialogAction('transcription'));
        dialogCancelBtn.addEventListener('click', closeActionDialog);

        // Initialize UI
        initializeSelectors();
        loadInitialJobs();
        setupDragAndDropListeners();
        updateFormatCounts();
    }

    function showLoginView() {
        if (appContainer) appContainer.style.display = 'none';
        if (loginContainer) loginContainer.style.display = 'flex';
        if (loginButton) loginButton.addEventListener('click', () => { window.location.href = apiUrl('/login'); });
    }

    // --- Entry Point ---
    if (window.APP_CONFIG && (window.APP_CONFIG.local_only_mode || window.APP_CONFIG.user)) {
        initializeApp();
    } else {
        showLoginView();
    }
});