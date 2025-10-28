document.addEventListener('DOMContentLoaded', () => {
    // --- Constants ---
    const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB chunks
    const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.api_base) ? window.APP_CONFIG.api_base.replace(/\/$/, '') : '';

    // --- User Locale ---
    const USER_LOCALE = navigator.language || 'en-US';
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
        // If WebSocket is connected, we don't need to poll
        if (webSocketConnected) {
            // Still check for any local state issues but don't poll server
            return;
        }
        
        // Fallback to polling if WebSocket is not connected
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
        // If WebSocket is connected, stop any existing polling
        if (webSocketConnected && jobPollerInterval) {
            clearInterval(jobPollerInterval);
            jobPollerInterval = null;
            console.log('Stopped polling since WebSocket is connected');
            return;
        }
        
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

    function renderJobRow(job) {
        const permanentDomId = `job-${job.id}`;
        let row = document.getElementById(permanentDomId);

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
        const formattedDate = new Date(job.created_at).toLocaleString(USER_LOCALE, DATETIME_FORMAT_OPTIONS);
        let statusHtml = `<span class="job-status-badge status-${job.status}">${job.status}</span>`;
        if ((job.status === 'processing' || job.status === 'pending') && job.task_type === 'unzip') {
            statusHtml += `<div class="progress-bar-container"><div class="progress-bar" style="width: ${job.progress || 0}%"></div></div>`;
        } else if (job.status === 'processing') {
            const progressClass = (job.progress > 0) ? '' : 'indeterminate';
            const progressWidth = (job.progress > 0) ? job.progress : 100;
            statusHtml += `<div class="progress-bar-container"><div class="progress-bar ${progressClass}" style="width: ${progressWidth}%"></div></div>`;
        }
        let actionHtml = '<span>-</span>';
        if (['pending', 'processing', 'uploading'].includes(job.status)) {
            actionHtml = `<button class="cancel-button" data-job-id="${job.id}"><i class="fa">&#xf00d;</i></button>`;
        } else if (job.status === 'completed') {
            if (job.task_type === 'unzip') {
                actionHtml = `<a href="${apiUrl('/download/zip-batch')}/${encodeURIComponent(job.id)}" class="download-button" download><i class="fa">&#xf019;</i> Batch</a>`;
            } else if (job.processed_filepath) {
                const downloadFilename = job.processed_filepath.split(/[\\\/]/).pop();
                actionHtml = `<a href="${apiUrl('/download')}/${encodeURIComponent(downloadFilename)}" class="download-button" download><i class="fa">&#xf019;</i></a>`;
            }
        } else if (job.status === 'failed') {
            const errorTitle = job.error_message ? ` title="${job.error_message.replace(/"/g, '&quot;')}"` : '';
            actionHtml = `<span class="error-text"${errorTitle}>Error</span>`;
        } else if (job.status === 'cancelled') {
            actionHtml = `<span>Cancelled</span>`;
        }
        let fileSizeHtml = job.input_filesize ? formatBytes(job.input_filesize) : '-';
        if (job.status === 'completed' && job.output_filesize) {
            fileSizeHtml += ` → ${formatBytes(job.output_filesize)}`;
        }
        let checkboxHtml = '';
        if (job.status === 'completed' && job.processed_filepath && job.task_type !== 'unzip') {
            checkboxHtml = `<input type="checkbox" class="job-checkbox" value="${job.id}">`;
        }

        // Truncate filename for mobile view
        const truncatedFilename = job.original_filename ? (job.original_filename.length > 25 ? job.original_filename.substring(0, 25) + '...' : job.original_filename) : "No filename";

        // --- Create or Update logic ---
        if (row) {
            // UPDATE an existing row
            const selectCell = row.querySelector('td[data-label="Select"] .cell-value');
            const fileCell = row.querySelector('td[data-label="File"] .cell-value');
            const taskCell = row.querySelector('td[data-label="Task"] .cell-value');
            const statusCell = row.querySelector('td[data-label="Status"] .cell-value');
            const actionCell = row.querySelector('td[data-label="Action"] .cell-value');
            
            if (selectCell) selectCell.innerHTML = checkboxHtml;
            // Calculate expanderHtml for the update section
            const expanderHtml = job.task_type === 'unzip' ? '<span class="expander-arrow"></span>' : '';
            if (fileCell) {
                fileCell.innerHTML = `<span class="file-cell-content" title="${job.original_filename}">${expanderHtml}${truncatedFilename}</span><button class="details-button" style="display: none;" title="Show details">i</button>`;
            }
            if (taskCell) taskCell.innerHTML = taskTypeLabel;
            if (statusCell) statusCell.innerHTML = statusHtml;
            if (actionCell) actionCell.innerHTML = actionHtml;
            
            // Update the expanded details if they exist
            const detailsRow = document.getElementById(`${permanentDomId}-details`);
            if (detailsRow) {
                detailsRow.querySelector('.details-full-filename').textContent = job.original_filename || "No filename";
                detailsRow.querySelector('.details-submitted').textContent = formattedDate;
                detailsRow.querySelector('.details-file-size').textContent = fileSizeHtml;
                detailsRow.querySelector('.details-id').textContent = job.id.substring(0, 8); // Truncate ID for display
                if (job.processed_filepath) {
                    const downloadLink = detailsRow.querySelector('.details-download-link');
                    if (downloadLink) {
                        const downloadFilename = job.processed_filepath.split(/[\\\/]/).pop();
                        downloadLink.href = `${apiUrl('/download')}/${encodeURIComponent(downloadFilename)}`;
                        downloadLink.textContent = downloadFilename;
                    }
                }
            }
        } else {
            // CREATE a new row
            row = document.createElement('tr');
            row.id = permanentDomId;
            // Sanitize filename to prevent XSS
            let escapedFilename = "No filename";
            if (job.original_filename) {
                escapedFilename = job.original_filename.replace(/[<>"'&]/g, function(match) {
                    return {
                        '<': '&lt;',
                        '>': '&gt;',
                        '"': '&quot;',
                        "'": '&#x27;',
                        '&': '&amp;'
                    }[match];
                });
            }
            const rowClasses = [];
            if (job.parent_job_id) rowClasses.push('sub-job');
            if (job.task_type === 'unzip') rowClasses.push('parent-job');
            row.className = rowClasses.join(' ');
            if (job.parent_job_id) row.dataset.parentId = job.parent_job_id;
            const expanderHtml = job.task_type === 'unzip' ? '<span class="expander-arrow"></span>' : '';

            // Define expanderHtml for the new row creation
            const newExpanderHtml = job.task_type === 'unzip' ? '<span class="expander-arrow"></span>' : '';
            // Create the row with all columns to match the table headers
            row.innerHTML = `
                <td data-label="Select"><span class="cell-value">${checkboxHtml}</span></td>
                <td data-label="File"><span class="cell-value" title="${escapedFilename}">${newExpanderHtml}<span class="file-cell-content">${truncatedFilename}</span><button class="details-button" style="display: none;" title="Show details">i</button></span></td>
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
            
            // Sanitize values for details
            const escapedOriginalFilename = job.original_filename ? job.original_filename.replace(/[<>"'&]/g, function(match) {
                return {
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#x27;',
                    '&': '&amp;'
                }[match];
            }) : "No filename";
            const escapedProcessedFilepath = job.processed_filepath ? job.processed_filepath.split(/[\\\/]/).pop() : "Not available";
            
            detailsRow.innerHTML = `
                <td colspan="5" class="job-details-content">
                    <div class="job-details-grid">
                        <div class="detail-item">
                            <span class="detail-label">Full Filename:</span>
                            <span class="detail-value details-full-filename">${escapedOriginalFilename}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Submitted:</span>
                            <span class="detail-value details-submitted">${formattedDate}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">File Size:</span>
                            <span class="detail-value details-file-size">${fileSizeHtml}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">ID:</span>
                            <span class="detail-value details-id">${job.id.substring(0, 8)}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Processed File:</span>
                            <span class="detail-value">${escapedProcessedFilepath}</span>
                        </div>
                        ${job.processed_filepath && job.status === 'completed' && job.task_type !== 'unzip' ? 
                            `<div class="detail-item">
                                <span class="detail-label">Download:</span>
                                <a class="detail-value details-download-link" href="${apiUrl('/download')}/${encodeURIComponent(job.processed_filepath.split(/[\\\/]/).pop())}" download>
                                    ${job.processed_filepath.split(/[\\\/]/).pop()}
                                </a>
                            </div>` : ''}
                        ${job.error_message ? 
                            `<div class="detail-item">
                                <span class="detail-label">Error:</span>
                                <span class="detail-value error-text details-error" title="${job.error_message.replace(/"/g, '&quot;')}">${job.error_message.length > 50 ? job.error_message.substring(0, 50) + '...' : job.error_message}</span>
                            </div>` : ''}
                    </div>
                </td>
            `;

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
                    const errorText = await response.text();
                    throw new Error(`Chunk upload failed: ${response.status} - ${errorText}`);
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
                    statusCell.innerHTML = `<span class="job-status-badge status-failed">Upload Failed</span>`;
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
                statusCell.innerHTML = `<span class="job-status-badge status-failed">Finalization Failed</span>`;
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

    function showActionDialog() {
        dialogFileCount.textContent = stagedFiles.length;
        
        if (stagedFiles.length === 1) {
            // If only one file is staged, update the format dropdown based on that file
            updateFormatsForFile(stagedFiles[0]).then(() => {
                // Initialize the dialog choices after updating formats
                if (dialogConversionChoices) dialogConversionChoices.destroy();
                dialogConversionChoices = new Choices(dialogOutputFormatSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a format...' });
                if (dialogTtsChoices) dialogTtsChoices.destroy();
                dialogTtsChoices = new Choices(dialogTtsModelSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a voice...' });
                dialogTtsChoices.setChoices(ttsModelsCache, 'value', 'label', true);
                dialogInitialView.style.display = 'grid';
                dialogConvertView.style.display = 'none';
                dialogTtsView.style.display = 'none';
                actionDialog.classList.add('visible');
            });
        } else {
            // If multiple files or no files, use all formats
            if (dialogConversionChoices) dialogConversionChoices.destroy();
            dialogConversionChoices = new Choices(dialogOutputFormatSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a format...' });
            // Set all available formats
            const tools = window.APP_CONFIG.conversionTools || {};
            const choicesArray = Object.keys(tools).map(toolKey => {
                const tool = tools[toolKey];
                return {
                    label: tool.name,
                    choices: Object.keys(tool.formats).map(formatKey => ({
                        value: `${toolKey}_${formatKey}`,
                        label: `${tool.name} - ${tool.formats[formatKey]}`
                    }))
                };
            });
            dialogConversionChoices.setChoices(choicesArray, 'value', 'label', true);
            
            if (dialogTtsChoices) dialogTtsChoices.destroy();
            dialogTtsChoices = new Choices(dialogTtsModelSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a voice...' });
            dialogTtsChoices.setChoices(ttsModelsCache, 'value', 'label', true);
            dialogInitialView.style.display = 'grid';
            dialogConvertView.style.display = 'none';
            dialogTtsView.style.display = 'none';
            actionDialog.classList.add('visible');
        }
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
        }
        Array.from(stagedFiles).forEach(file => uploadFileInChunks(file, action, options));
        closeActionDialog();
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
    const tools = window.APP_CONFIG.conversionTools || {};
    const choicesArray = Object.keys(tools).map(toolKey => {
        const tool = tools[toolKey];
        return {
            label: tool.name,
            choices: Object.keys(tool.formats).map(formatKey => ({
                value: `${toolKey}_${formatKey}`,
                // --- THIS IS THE MODIFIED LINE ---
                label: `${tool.name} - ${tool.formats[formatKey]}`
            }))
        };
    });
    conversionChoices.setChoices(choicesArray, 'value', 'label', true);

    if (transcriptionChoices) transcriptionChoices.destroy();
    transcriptionChoices = new Choices(mainModelSizeSelect, { searchEnabled: false, shouldSort: false, itemSelectText: '' });

    if (ttsChoices) ttsChoices.destroy();
    ttsChoices = new Choices(mainTtsModelSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select voice...' });
    loadTtsModels();
}

    function getFileExtension(filename) {
        return '.' + filename.split('.').pop().toLowerCase();
    }

    async function updateFormatsForFile(file) {
        if (!file) return;
        
        const fileExtension = getFileExtension(file.name);
        try {
            const response = await authFetch(`/api/v1/supported-formats/${encodeURIComponent(fileExtension)}`);
            if (!response.ok) {
                console.error(`Failed to fetch supported formats for ${fileExtension}:`, response.status);
                return;
            }
            
            const data = await response.json();
            const formats = data.formats || [];
            
            // Update main output format select
            if (conversionChoices) {
                // Clear existing choices
                conversionChoices.clearStore();
                
                // Group formats by tool name for better UI
                const groupedFormats = formats.reduce((acc, format) => {
                    if (!acc[format.tool]) {
                        acc[format.tool] = {
                            label: window.APP_CONFIG.conversionTools[format.tool]?.name || format.tool,
                            choices: []
                        };
                    }
                    acc[format.tool].choices.push({
                        value: format.value,
                        label: format.label
                    });
                    return acc;
                }, {});
                
                // Convert grouped formats to choices array
                const choicesArray = Object.keys(groupedFormats).map(toolKey => ({
                    label: groupedFormats[toolKey].label,
                    choices: groupedFormats[toolKey].choices
                }));
                
                conversionChoices.setChoices(choicesArray, 'value', 'label', true);
            }
            
            // Update dialog output format select
            if (dialogConversionChoices) {
                // Clear existing choices
                dialogConversionChoices.clearStore();
                
                // Group formats by tool name for better UI
                const dialogGroupedFormats = formats.reduce((acc, format) => {
                    if (!acc[format.tool]) {
                        acc[format.tool] = {
                            label: window.APP_CONFIG.conversionTools[format.tool]?.name || format.tool,
                            choices: []
                        };
                    }
                    acc[format.tool].choices.push({
                        value: format.value,
                        label: format.label
                    });
                    return acc;
                }, {});
                
                // Convert grouped formats to choices array
                const dialogChoicesArray = Object.keys(dialogGroupedFormats).map(toolKey => ({
                    label: dialogGroupedFormats[toolKey].label,
                    choices: dialogGroupedFormats[toolKey].choices
                }));
                
                dialogConversionChoices.setChoices(dialogChoicesArray, 'value', 'label', true);
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
            updateFormatsForFile(input.files[0]);
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
            const response = await authFetch(`/job/${jobId}/cancel`, { method: 'POST' });
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
        downloadSelectedBtn.disabled = selectedCheckboxes.length === 0;
        selectAllJobsCheckbox.checked = jobListBody.querySelectorAll('.job-checkbox').length > 0 && selectedCheckboxes.length === jobListBody.querySelectorAll('.job-checkbox').length;
    }

    async function handleBatchDownload() {
        const selectedIds = Array.from(jobListBody.querySelectorAll('.job-checkbox:checked')).map(cb => cb.value);
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
                    
                    // Now we need to update the details with actual job data by finding the job object
                    // We'll do this by looking for the job in the DOM or by re-rendering
                    const jobData = Array.from(jobListBody.querySelectorAll('tr[id^="job-"]'))
                        .filter(tr => tr.id !== `job-${jobId}-details`)  // Exclude details rows
                        .map(tr => {
                            const jobId = tr.id.replace('job-', '');
                            // We need to get the job data - this is tricky without a lookup table
                            // For now, we'll update all job rows which will update the details
                            return jobId;
                        });
                    
                    // Trigger a refresh of the job details by re-rendering the job
                    // We'll need to get the job data from the server
                    authFetch(`/job/${jobId}`)
                        .then(response => response.json())
                        .then(job => {
                            const formattedDate = new Date(job.created_at).toLocaleString(USER_LOCALE, DATETIME_FORMAT_OPTIONS);
                            let fileSizeHtml = job.input_filesize ? formatBytes(job.input_filesize) : '-';
                            if (job.status === 'completed' && job.output_filesize) {
                                fileSizeHtml += ` → ${formatBytes(job.output_filesize)}`;
                            }
                            
                            // Sanitize values for details
                            const escapedOriginalFilename = job.original_filename ? job.original_filename.replace(/[<>"'&]/g, function(match) {
                                return {
                                    '<': '&lt;',
                                    '>': '&gt;',
                                    '"': '&quot;',
                                    "'": '&#x27;',
                                    '&': '&amp;'
                                }[match];
                            }) : "No filename";
                            const escapedProcessedFilepath = job.processed_filepath ? job.processed_filepath.split(/[\\\/]/).pop() : "Not available";
                            
                            detailsRow.innerHTML = `
                                <td colspan="7" class="job-details-content">
                                    <div class="job-details-grid">
                                        <div class="detail-item">
                                            <span class="detail-label">Full Filename:</span>
                                            <span class="detail-value details-full-filename">${escapedOriginalFilename}</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-label">Submitted:</span>
                                            <span class="detail-value details-submitted">${formattedDate}</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-label">File Size:</span>
                                            <span class="detail-value details-file-size">${fileSizeHtml}</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-label">ID:</span>
                                            <span class="detail-value details-id">${job.id.substring(0, 8)}</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-label">Processed File:</span>
                                            <span class="detail-value">${escapedProcessedFilepath}</span>
                                        </div>
                                        ${job.processed_filepath && job.status === 'completed' && job.task_type !== 'unzip' ? 
                                            `<div class="detail-item">
                                                <span class="detail-label">Download:</span>
                                                <a class="detail-value details-download-link" href="${apiUrl('/download')}/${encodeURIComponent(job.processed_filepath.split(/[\\\/]/).pop())}" download>
                                                    ${job.processed_filepath.split(/[\\\/]/).pop()}
                                                </a>
                                            </div>` : ''}
                                        ${job.error_message ? 
                                            `<div class="detail-item">
                                                <span class="detail-label">Error:</span>
                                                <span class="detail-value error-text details-error" title="${job.error_message.replace(/"/g, '&quot;')}">${job.error_message.length > 50 ? job.error_message.substring(0, 50) + '...' : job.error_message}</span>
                                            </div>` : ''}
                                    </div>
                                </td>
                            `;
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
    
    // --- WebSocket Manager ---
    const ENABLE_WEBSOCKETS = false;
    let webSocket = null;
    let webSocketConnected = false;
    let isUsingPollingFallback = false;
    let reconnectAttempts = 0;
    const RECONNECT_INTERVAL = 3000; // 3 seconds

    function initializeWebSocket() {
        // Close existing connection if any
        if (webSocket) {
            webSocket.close();
        }

        // Construct WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/jobs`;
        
        // For authentication, add token if available (in a production setup)
        const fullUrl = wsUrl; // For now, no token needed since we're in local mode

        try {
            webSocket = new WebSocket(fullUrl);
            
            webSocket.onopen = function(event) {
                console.log('WebSocket connected');
                webSocketConnected = true;
                isUsingPollingFallback = false;
                reconnectAttempts = 0; // Reset on successful connection
                // Start a heartbeat to keep connection alive
                startHeartbeat();
                // Update polling behavior since WebSocket is connected
                if (jobPollerInterval) {
                    clearInterval(jobPollerInterval);
                    jobPollerInterval = null;
                    console.log('Stopped polling since WebSocket is connected');
                }
            };

            webSocket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            webSocket.onclose = function(event) {
                console.log('WebSocket disconnected:', event.code, event.reason);
                webSocketConnected = false;
                isUsingPollingFallback = true;
                // Attempt to reconnect
                attemptReconnect();
                // Start polling as a fallback
                startJobPolling();
            };

            webSocket.onerror = function(error) {
                console.error('WebSocket error:', error);
                webSocketConnected = false;
            };
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            attemptReconnect();
        }
    }

    function handleWebSocketMessage(data) {
        switch (data.type) {
            case 'connection_established':
                console.log('WebSocket connection established for user:', data.user_id);
                updateConnectionStatus(true);
                break;
                
            case 'job_update':
                // Update a single job
                if (data.job) {
                    const jobId = data.job.id;
                    console.log('Received job update:', jobId, data.job.status, data.job.progress);
                    renderJobRow(data.job);
                }
                break;
                
            case 'batch_job_update':
                // Update multiple jobs
                if (data.jobs && Array.isArray(data.jobs)) {
                    data.jobs.forEach(job => {
                        renderJobRow(job);
                    });
                }
                break;
                
            case 'pong':
                // Heartbeat response
                break;
                
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }

    function attemptReconnect() {
        reconnectAttempts++;
        const delay = Math.min(RECONNECT_INTERVAL * Math.pow(2, reconnectAttempts), 30000); // Exponential backoff up to 30 seconds
        console.log(`Attempting to reconnect... (attempt ${reconnectAttempts}, next in ${delay / 1000}s)`);
        setTimeout(initializeWebSocket, delay);
    }

    let heartbeatInterval;

    function startHeartbeat() {
        // Clear existing interval
        if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
        }
        
        // Send ping every 30 seconds to keep connection alive
        heartbeatInterval = setInterval(() => {
            if (webSocket && webSocket.readyState === WebSocket.OPEN) {
                webSocket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // 30 seconds
    }

    function stopHeartbeat() {
        if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
        }
    }




    // Connection status UI updates
    function updateConnectionStatus(connected) {
        const statusDot = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        
        if (statusDot && statusText) {
            if (connected) {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Real-time connection';
            } else {
                statusDot.className = 'status-dot disconnected';
                statusText.textContent = 'Polling active';
            }
        }
    }
    
    // Close WebSocket when page unloads
    window.addEventListener('beforeunload', () => {
        if (webSocket) {
            stopHeartbeat();
            webSocket.close();
        }
    });
    
    // Initialize WebSocket connection after UI is set up
    if (ENABLE_WEBSOCKETS && window.APP_CONFIG && (window.APP_CONFIG.local_only_mode || window.APP_CONFIG.user)) {
        initializeWebSocket();
    }
});