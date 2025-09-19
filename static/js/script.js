document.addEventListener('DOMContentLoaded', () => {
    // --- Constants ---
    const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB chunks

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
    const startConversionBtn = document.getElementById('start-conversion-btn');
    const startOcrBtn = document.getElementById('start-ocr-btn');
    const startTranscriptionBtn = document.getElementById('start-transcription-btn');

    const jobListBody = document.getElementById('job-list-body');

    // Drag and Drop Elements
    const dragOverlay = document.getElementById('drag-overlay');
    const actionDialog = document.getElementById('action-dialog');
    const dialogFileCount = document.getElementById('dialog-file-count');
    const dialogInitialView = document.getElementById('dialog-initial-actions');
    const dialogConvertView = document.getElementById('dialog-convert-view');
    const dialogConvertBtn = document.getElementById('dialog-action-convert');
    const dialogOcrBtn = document.getElementById('dialog-action-ocr');
    const dialogTranscribeBtn = document.getElementById('dialog-action-transcribe');
    const dialogCancelBtn = document.getElementById('dialog-action-cancel');
    const dialogStartConversionBtn = document.getElementById('dialog-start-conversion');
    const dialogBackBtn = document.getElementById('dialog-back');
    const dialogOutputFormatSelect = document.getElementById('dialog-output-format-select');

    // --- State Variables ---
    let conversionChoices = null;
    let modelChoices = null; // For the model dropdown instance
    let dialogConversionChoices = null;
    const activePolls = new Map();
    let stagedFiles = null;


    // --- Authentication-aware Fetch Wrapper ---
    /**
     * A wrapper around the native fetch API that handles 401 Unauthorized responses.
     * If a 401 is received, it assumes the session has expired and redirects to the login page.
     * @param {string} url - The URL to fetch.
     * @param {object} options - The options for the fetch request.
     * @returns {Promise<Response>} - A promise that resolves to the fetch Response.
     */
    async function authFetch(url, options) {
        const response = await fetch(url, options);
        if (response.status === 401) {
            // Use a simple alert for now. A more sophisticated modal could be used.
            alert('Your session has expired. You will be redirected to the login page.');
            window.location.href = '/login';
            // Throw an error to stop the promise chain of the calling function
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
                const errorData = await finalizeResponse.json();
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
            options.model_size = mainModelSizeSelect.value;
        }

        // Disable buttons during upload process
        startConversionBtn.disabled = true;
        startOcrBtn.disabled = true;
        startTranscriptionBtn.disabled = true;

        const uploadPromises = files.map(file => uploadFileInChunks(file, taskType, options));
        await Promise.allSettled(uploadPromises);

        // Reset file input and re-enable buttons
        mainFileInput.value = ''; // Resets the file list
        updateFileName(mainFileInput, mainFileName);
        startConversionBtn.disabled = false;
        startOcrBtn.disabled = false;
        startTranscriptionBtn.disabled = false;
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
        dialogOutputFormatSelect.innerHTML = mainOutputFormatSelect.innerHTML; // Use main select as template
        if (dialogConversionChoices) dialogConversionChoices.destroy();
        dialogConversionChoices = new Choices(dialogOutputFormatSelect, {
            searchEnabled: true,
            itemSelectText: 'Select',
            shouldSort: false,
            placeholder: true,
            placeholderValue: 'Select a format...',
        });
        dialogInitialView.style.display = 'grid';
        dialogConvertView.style.display = 'none';
        actionDialog.classList.add('visible');
    }

    function closeActionDialog() {
        actionDialog.classList.remove('visible');
        stagedFiles = null;
        if (dialogConversionChoices) {
            dialogConversionChoices.hideDropdown();
            dialogConversionChoices.destroy();
            dialogConversionChoices = null;
        }
    }

    dialogConvertBtn.addEventListener('click', () => {
        dialogInitialView.style.display = 'none';
        dialogConvertView.style.display = 'block';
    });
    dialogBackBtn.addEventListener('click', () => {
        dialogInitialView.style.display = 'grid';
        dialogConvertView.style.display = 'none';
    });
    dialogStartConversionBtn.addEventListener('click', () => handleDialogAction('conversion'));
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
        }
        Array.from(stagedFiles).forEach(file => uploadFileInChunks(file, action, options));
        closeActionDialog();
    }

    /**
     * Initializes all Choices.js dropdowns on the page.
     */
    function initializeSelectors() {
        // --- Conversion Dropdown ---
        if (conversionChoices) conversionChoices.destroy();
        conversionChoices = new Choices(mainOutputFormatSelect, {
            searchEnabled: true,
            itemSelectText: 'Select',
            shouldSort: false,
            placeholder: true,
            placeholderValue: 'Select a format...',
        });
        const tools = window.APP_CONFIG.conversionTools || {};
        const choicesArray = [];
        for (const toolKey in tools) {
            const tool = tools[toolKey];
            const group = { label: tool.name, id: toolKey, disabled: false, choices: [] };
            for (const formatKey in tool.formats) {
                group.choices.push({
                    value: `${toolKey}_${formatKey}`,
                    label: `${tool.name} - ${formatKey.toUpperCase()} (${tool.formats[formatKey]})`
                });
            }
            choicesArray.push(group);
        }
        conversionChoices.setChoices(choicesArray, 'value', 'label', true);

        // --- Model Size Dropdown ---
        if (modelChoices) modelChoices.destroy();
        modelChoices = new Choices(mainModelSizeSelect, {
            searchEnabled: false,   // Disables the search box
            shouldSort: false,      // Keeps the original <option> order
            itemSelectText: '',     // Hides the "Press to select" tooltip
        });
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
                const errorData = await response.json();
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

    async function loadInitialJobs() {
        try {
            const response = await authFetch('/jobs');
            if (!response.ok) throw new Error('Failed to fetch jobs.');
            const jobs = await response.json();
            jobListBody.innerHTML = '';
            for (const job of jobs.reverse()) {
                renderJobRow(job);
                if (['pending', 'processing'].includes(job.status)) startPolling(job.id);
            }
        } catch (error) {
            console.error("Couldn't load job history:", error);
            if (error.message !== 'Session expired') {
                jobListBody.innerHTML = '<tr><td colspan="6" style="text-align: center;">Could not load job history.</td></tr>';
            }
        }
    }

    function startPolling(jobId) {
        if (activePolls.has(jobId)) return;
        const intervalId = setInterval(async () => {
            try {
                const response = await authFetch(`/job/${jobId}`);
                if (!response.ok) {
                    if (response.status === 404) stopPolling(jobId);
                    return;
                }
                const job = await response.json();
                renderJobRow(job);
                if (['completed', 'failed', 'cancelled'].includes(job.status)) stopPolling(jobId);
            } catch (error) {
                console.error(`Error polling for job ${jobId}:`, error);
                stopPolling(jobId); // Stop polling on any error, including auth errors
            }
        }, 2500);
        activePolls.set(jobId, intervalId);
    }

    function stopPolling(jobId) {
        if (activePolls.has(jobId)) {
            clearInterval(activePolls.get(jobId));
            activePolls.delete(jobId);
        }
    }

    function renderJobRow(job) {
        const rowId = job.id.startsWith('upload-') ? job.id : `job-${job.id}`;
        let row = document.getElementById(rowId);
        if (!row) {
            row = document.createElement('tr');
            row.id = rowId;
            jobListBody.prepend(row);
        }

        let taskTypeLabel = job.task_type;
        if (job.task_type === 'conversion' && job.processed_filepath) {
            const extension = job.processed_filepath.split('.').pop();
            taskTypeLabel = `Convert to ${extension.toUpperCase()}`;
        }

        const submittedDate = new Date(job.created_at);
        const formattedDate = submittedDate.toLocaleString(USER_LOCALE, DATETIME_FORMAT_OPTIONS);

        let statusHtml = `<span class="job-status-badge status-${job.status}">${job.status}</span>`;
        if (job.status === 'uploading') {
            statusHtml = `<span class="job-status-badge status-processing">Uploading</span>`;
            statusHtml += `<div class="progress-bar-container"><div class="progress-bar" style="width: ${job.progress || 0}%"></div></div>`;
        } else if (job.status === 'processing') {
            const progressClass = (job.task_type === 'transcription' && job.progress > 0) ? '' : 'indeterminate';
            const progressWidth = job.task_type === 'transcription' ? job.progress : 100;
            statusHtml += `<div class="progress-bar-container"><div class="progress-bar ${progressClass}" style="width: ${progressWidth}%"></div></div>`;
        }

        let actionHtml = `<span>-</span>`;
        if (['pending', 'processing'].includes(job.status)) {
            actionHtml = `<button class="cancel-button" data-job-id="${job.id}">Cancel</button>`;
        } else if (job.status === 'completed' && job.processed_filepath) {
            const downloadFilename = job.processed_filepath.split(/[\\/]/).pop();
            actionHtml = `<a href="/download/${downloadFilename}" class="download-button" download>Download</a>`;
        } else if (job.status === 'failed') {
            const errorTitle = job.error_message ? ` title="${job.error_message.replace(/"/g, '&quot;')}"` : '';
            actionHtml = `<span class="error-text"${errorTitle}>Failed</span>`;
        }

        let fileSizeHtml = job.input_filesize ? formatBytes(job.input_filesize) : '-';
        if (job.status === 'completed' && job.output_filesize) {
            fileSizeHtml += ` → ${formatBytes(job.output_filesize)}`;
        }

        const escapedFilename = job.original_filename ? job.original_filename.replace(/</g, "&lt;").replace(/>/g, "&gt;") : "No filename";

        row.innerHTML = `
            <td data-label="File"><span class="cell-value" title="${escapedFilename}">${escapedFilename}</span></td>
            <td data-label="File Size"><span class="cell-value">${fileSizeHtml}</span></td>
            <td data-label="Task"><span class="cell-value">${taskTypeLabel}</span></td>
            <td data-label="Submitted"><span class="cell-value">${formattedDate}</span></td>
            <td data-label="Status"><span class="cell-value">${statusHtml}</span></td>
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
        mainFileInput.addEventListener('change', () => updateFileName(mainFileInput, mainFileName));

        jobListBody.addEventListener('click', (event) => {
            if (event.target.classList.contains('cancel-button')) {
                const jobId = event.target.dataset.jobId;
                handleCancelJob(jobId);
            }
        });

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
                window.location.href = '/login';
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