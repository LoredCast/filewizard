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
    const CHUNK_ATTEMPTS = 4; // a chunk is retried after network errors and 5xx responses
    const MAX_PARALLEL_UPLOADS = 3; // more files wait in a queue, so the first ones finish (and start processing) sooner
    const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.api_base) ? window.APP_CONFIG.api_base.replace(/\/$/, '') : '';
    const UPLOAD_LIMITS = window.APP_CONFIG.uploadLimits || {};
    const ACTIVE_STATUSES = new Set(['pending', 'processing']);
    const FINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);

    // Polling: only jobs changed since the newest update seen so far are fetched. The overlap
    // re-fetches the last few seconds, so a job whose update was committed late is not missed.
    const POLL_MIN_MS = 1500;
    const POLL_MAX_MS = 6000; // the interval grows towards this while nothing changes
    const POLL_ERROR_MAX_MS = 30000;
    const POLL_OVERLAP_MS = 15000;

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
    let sessionExpired = false;

    // Job history: id -> { job, signature } of every rendered job.
    const renderedJobs = new Map();
    let newestUpdateMs = null; // newest updated_at seen, the cursor for incremental polling
    let pollTimer = null;
    let pollInFlight = false;
    let pollDelay = POLL_MIN_MS;
    let pollErrors = 0;

    // Uploads: waiting entries and the number currently transferring.
    const uploadQueue = [];
    const uploadsById = new Map();
    let activeUploads = 0;

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
            if (!sessionExpired) { // parallel requests would otherwise each show the alert
                sessionExpired = true;
                alert('Your session has expired. You will be redirected to the login page.');
                window.location.href = apiUrl('/login');
            }
            throw new Error('Session expired');
        }
        return response;
    }

    async function errorDetail(response, fallback) {
        const data = await response.json().catch(() => ({}));
        return (typeof data.detail === 'string' && data.detail) || fallback || `Request failed (HTTP ${response.status})`;
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function formatBytes(bytes, decimals = 1) {
        if (!+bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    // Escape a value for safe interpolation into HTML text or a quoted attribute.
    // Every server- or user-controlled string (file names, tool error output, ...) must go through this.
    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, ch => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[ch]));
    }

    function getFileExtension(filename) {
        const dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.slice(dot).toLowerCase() : '';
    }

    // --- Connection status (bottom bar) ---
    function updateConnectionStatus() {
        const statusDot = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        if (!statusDot || !statusText) return;
        let state = 'idle', text = 'Up to date';
        if (pollErrors > 0) {
            state = 'error'; text = 'Connection lost – retrying…';
        } else if (hasActiveJobs()) {
            state = 'connected'; text = document.hidden ? 'Paused while in background' : 'Live updates';
        }
        statusDot.className = `status-dot ${state}`;
        statusText.textContent = text;
    }

    // --- Job polling ---
    function hasActiveJobs() {
        for (const { job } of renderedJobs.values()) {
            if (ACTIVE_STATUSES.has(job.status)) return true;
        }
        return false;
    }

    function schedulePoll(delay) {
        clearTimeout(pollTimer);
        pollTimer = null;
        updateConnectionStatus();
        // Nothing to watch, or the tab is hidden (polling resumes when it becomes visible again).
        if (pollInFlight || document.hidden || (!hasActiveJobs() && pollErrors === 0)) return;
        pollTimer = setTimeout(pollForJobUpdates, delay);
    }

    // Poll soon, e.g. after a job was submitted or cancelled.
    function requestPoll(delay = POLL_MIN_MS) {
        pollDelay = POLL_MIN_MS;
        schedulePoll(delay);
    }

    async function pollForJobUpdates() {
        pollTimer = null;
        if (pollInFlight) return;
        pollInFlight = true;
        let changed = 0;
        try {
            const url = newestUpdateMs === null
                ? '/jobs'
                : `/jobs?since=${encodeURIComponent(new Date(newestUpdateMs - POLL_OVERLAP_MS).toISOString())}`;
            const response = await authFetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            changed = renderJobs(await response.json());
            pollErrors = 0;
        } catch (error) {
            if (error.message === 'Session expired') {
                pollInFlight = false;
                return;
            }
            pollErrors++;
            console.error('Job polling failed:', error);
        } finally {
            pollInFlight = false;
        }
        if (pollErrors > 0) {
            pollDelay = Math.min(POLL_ERROR_MAX_MS, POLL_MIN_MS * 2 ** pollErrors);
        } else {
            pollDelay = changed > 0 ? POLL_MIN_MS : Math.min(POLL_MAX_MS, Math.round(pollDelay * 1.5));
        }
        schedulePoll(pollDelay);
    }

    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) requestPoll(0); // catch up right away
        else updateConnectionStatus();
    });

    // --- Job rendering ---
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

    function jobSignature(job) {
        return [job.status, job.progress, job.updated_at, job.error_message, job.output_filesize, job.processed_filepath].join('|');
    }

    function taskTypeLabel(job) {
        if (job.task_type === 'conversion' && job.processed_filepath) {
            return `Convert to ${job.processed_filepath.split('.').pop().toUpperCase()}`;
        }
        const labels = { academic_pandoc: 'Academic PDF', tts: 'Synthesize Speech', unzip: 'Unpack ZIP' };
        if (labels[job.task_type]) return labels[job.task_type];
        return job.task_type ? job.task_type.charAt(0).toUpperCase() + job.task_type.slice(1) : '';
    }

    // Progress bar markup; `indeterminate` animates while a job reports no progress yet.
    function progressBarHtml(progress, indeterminate = false) {
        return `<div class="progress-bar-container"><div class="progress-bar${indeterminate ? ' indeterminate' : ''}" style="width: ${indeterminate ? 100 : progress}%"></div></div>`;
    }

    function statusCellHtml(job) {
        const status = escapeHtml(job.status);
        const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
        let html = `<span class="job-status-badge status-${status}">${status}</span>`;
        if (ACTIVE_STATUSES.has(job.status) && job.task_type === 'unzip') {
            html += progressBarHtml(progress);
        } else if (job.status === 'processing') {
            html += progressBarHtml(progress, progress === 0);
        }
        return html;
    }

    function actionCellHtml(job) {
        const jobId = escapeHtml(job.id);
        if (ACTIVE_STATUSES.has(job.status)) {
            return `<button class="cancel-button" data-job-id="${jobId}" title="Cancel"><i class="fa">&#xf00d;</i></button>`;
        }
        if (job.status === 'completed') {
            if (job.task_type === 'unzip') {
                return `<a href="${escapeHtml(apiUrl('/download/zip-batch') + '/' + encodeURIComponent(job.id))}" class="download-button" download><i class="fa">&#xf019;</i> Batch</a>`;
            }
            if (job.processed_filepath) {
                return `<a href="${escapeHtml(apiUrl('/download') + '/' + encodeURIComponent(processedBasename(job)))}" class="download-button" download><i class="fa">&#xf019;</i></a>`;
            }
        } else if (job.status === 'failed') {
            const errorTitle = job.error_message ? ` title="${escapeHtml(job.error_message)}"` : '';
            return `<span class="error-text"${errorTitle}>Error</span>`;
        } else if (job.status === 'cancelled') {
            return '<span>Cancelled</span>';
        }
        return '<span>-</span>';
    }

    // Every finished job can be selected (for deletion); only jobs with a result file can be downloaded.
    function checkboxHtml(job) {
        if (!FINAL_STATUSES.has(job.status)) return '';
        const downloadable = job.status === 'completed' && job.processed_filepath && job.task_type !== 'unzip';
        return `<input type="checkbox" class="job-checkbox" value="${escapeHtml(job.id)}"${downloadable ? ' data-downloadable="1"' : ''}>`;
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

    function cellValue(row, label) {
        return row.querySelector(`td[data-label="${label}"] .cell-value`);
    }

    function createJobRows(job) {
        const row = document.createElement('tr');
        row.id = `job-${job.id}`;
        if (job.parent_job_id) {
            row.classList.add('sub-job');
            row.dataset.parentId = job.parent_job_id;
        }
        if (job.task_type === 'unzip') row.classList.add('parent-job');

        // Truncate filename for mobile view (truncate first, then escape, so entities are never cut in half)
        const rawFilename = job.original_filename || 'No filename';
        const escapedFilename = escapeHtml(rawFilename);
        const truncatedFilename = escapeHtml(rawFilename.length > 25 ? rawFilename.substring(0, 25) + '...' : rawFilename);
        const expanderHtml = job.task_type === 'unzip' ? '<span class="expander-arrow"></span>' : '';
        row.innerHTML = `
            <td data-label="Select"><span class="cell-value">${checkboxHtml(job)}</span></td>
            <td data-label="File"><span class="cell-value" title="${escapedFilename}">${expanderHtml}<span class="file-cell-content">${truncatedFilename}</span><button class="details-button" style="display: none;" title="Show details">i</button></span></td>
            <td data-label="File Size"><span class="cell-value">${escapeHtml(formatJobFileSize(job))}</span></td>
            <td data-label="Task"><span class="cell-value">${escapeHtml(taskTypeLabel(job))}</span></td>
            <td data-label="Submitted"><span class="cell-value">${escapeHtml(formatJobDate(job))}</span></td>
            <td data-label="Status"><span class="cell-value status-cell-value">${statusCellHtml(job)}</span></td>
            <td data-label="Action" class="action-col"><span class="cell-value">${actionCellHtml(job)}</span></td>
        `;

        const detailsRow = document.createElement('tr');
        detailsRow.id = `job-${job.id}-details`;
        detailsRow.className = 'job-details-row';
        detailsRow.style.display = 'none';
        if (job.parent_job_id) detailsRow.dataset.parentId = job.parent_job_id;
        detailsRow.innerHTML = buildDetailsHtml(job);

        const parentRow = job.parent_job_id ? document.getElementById(`job-${job.parent_job_id}`) : null;
        if (parentRow) {
            // Sub-jobs go below their batch (after its details row and earlier sub-jobs), in submission order.
            let anchor = document.getElementById(`${parentRow.id}-details`) || parentRow;
            while (anchor.nextElementSibling && anchor.nextElementSibling.dataset.parentId === job.parent_job_id) {
                anchor = anchor.nextElementSibling;
            }
            anchor.after(row, detailsRow);
            if (parentRow.classList.contains('sub-jobs-visible')) row.classList.add('is-visible');
        } else {
            jobListBody.prepend(row, detailsRow);
        }
    }

    // Updates only the cells whose content changed, so selections, open details and the
    // progress bar's transition survive a poll.
    function updateJobRows(row, job, previous) {
        const selectCell = cellValue(row, 'Select');
        const newCheckbox = checkboxHtml(job);
        if (selectCell && newCheckbox !== checkboxHtml(previous)) {
            const wasChecked = selectCell.querySelector('.job-checkbox')?.checked;
            selectCell.innerHTML = newCheckbox;
            const checkbox = selectCell.querySelector('.job-checkbox');
            if (checkbox && wasChecked) checkbox.checked = true;
        }
        const sizeCell = cellValue(row, 'File Size');
        if (sizeCell) sizeCell.textContent = formatJobFileSize(job);
        const taskCell = cellValue(row, 'Task');
        if (taskCell) taskCell.textContent = taskTypeLabel(job);

        const statusCell = cellValue(row, 'Status');
        const newStatusHtml = statusCellHtml(job);
        const bar = statusCell && statusCell.querySelector('.progress-bar:not(.indeterminate)');
        if (bar && job.status === previous.status && Number(job.progress) > 0 && !newStatusHtml.includes('indeterminate')) {
            bar.style.width = `${Math.max(0, Math.min(100, Number(job.progress) || 0))}%`; // animates via CSS
        } else if (statusCell) {
            statusCell.innerHTML = newStatusHtml;
        }
        const actionCell = cellValue(row, 'Action');
        const newActionHtml = actionCellHtml(job);
        if (actionCell && newActionHtml !== actionCellHtml(previous)) actionCell.innerHTML = newActionHtml;

        const detailsRow = document.getElementById(`${row.id}-details`);
        if (detailsRow) detailsRow.innerHTML = buildDetailsHtml(job);
    }

    // Renders a job, creating or updating its rows. Returns true if anything changed.
    function renderJobRow(job) {
        if (!job || !job.id) return false;
        const updatedMs = Date.parse(job.updated_at);
        if (!Number.isNaN(updatedMs) && (newestUpdateMs === null || updatedMs > newestUpdateMs)) newestUpdateMs = updatedMs;

        const signature = jobSignature(job);
        const known = renderedJobs.get(job.id);
        const row = document.getElementById(`job-${job.id}`);
        if (known && row) {
            if (known.signature === signature) return false;
            updateJobRows(row, job, known.job);
        } else {
            createJobRows(job);
        }
        renderedJobs.set(job.id, { job, signature });
        return true;
    }

    // Renders a list of jobs (oldest first, so batches exist before their sub-jobs). Returns the number changed.
    function renderJobs(jobs) {
        const sorted = [...jobs].sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at));
        let changed = 0;
        for (const job of sorted) {
            if (renderJobRow(job)) changed++;
        }
        if (changed) handleSelectionChange();
        return changed;
    }

    function removeJobRows(jobId) {
        document.getElementById(`job-${jobId}`)?.remove();
        document.getElementById(`job-${jobId}-details`)?.remove();
        renderedJobs.delete(jobId);
    }

    // --- Uploads ---
    // Returns why the server would refuse this file for the task, or '' if it looks fine.
    function uploadProblem(file, taskType) {
        const extension = getFileExtension(file.name);
        const maxBytes = Number(UPLOAD_LIMITS.max_file_size_bytes) || 0;
        if (maxBytes && file.size > maxBytes) {
            return `File is too large (${formatBytes(file.size)}; the limit is ${formatBytes(maxBytes)}).`;
        }
        const allowed = UPLOAD_LIMITS.allowed_extensions || [];
        if (allowed.length && !allowed.includes(extension)) {
            return `Files of type '${extension || file.name}' are not allowed on this server.`;
        }
        const ocrExtensions = UPLOAD_LIMITS.ocr_extensions || [];
        if (taskType === 'ocr' && ocrExtensions.length && extension !== '.zip' && !ocrExtensions.includes(extension)) {
            return `OCR needs a PDF or an image (${ocrExtensions.join(', ')}).`;
        }
        return '';
    }

    function createUploadRow(file, taskType, status) {
        const row = document.createElement('tr');
        row.id = `upload-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
        row.className = 'upload-row';
        const escapedFilename = escapeHtml(file.name);
        const taskLabel = escapeHtml(taskType.charAt(0).toUpperCase() + taskType.slice(1));
        row.innerHTML = `
            <td data-label="Select"><span class="cell-value">-</span></td>
            <td data-label="File"><span class="cell-value" title="${escapedFilename}">${escapedFilename}</span></td>
            <td data-label="File Size"><span class="cell-value">${escapeHtml(formatBytes(file.size))}</span></td>
            <td data-label="Task"><span class="cell-value">${taskLabel}</span></td>
            <td data-label="Submitted"><span class="cell-value">${escapeHtml(new Date().toLocaleString(USER_LOCALE, DATETIME_FORMAT_OPTIONS))}</span></td>
            <td data-label="Status"><span class="cell-value status-cell-value"><span class="job-status-badge status-${status}">${status}</span></span></td>
            <td data-label="Action" class="action-col"><span class="cell-value"><button class="cancel-button" data-upload-id="${row.id}" title="Cancel upload"><i class="fa">&#xf00d;</i></button></span></td>
        `;
        jobListBody.prepend(row);
        return row;
    }

    function showUploadFailure(row, label, message) {
        const statusCell = row.querySelector('.status-cell-value');
        if (statusCell) statusCell.innerHTML = `<span class="job-status-badge status-failed" title="${escapeHtml(message)}">${escapeHtml(label)}</span>`;
        const actionCell = cellValue(row, 'Action');
        if (actionCell) {
            actionCell.innerHTML = `<span class="error-text" title="${escapeHtml(message)}">Error</span> <button class="cancel-button" data-dismiss-upload="${row.id}" title="Remove from list"><i class="fa">&#xf00d;</i></button>`;
        }
    }

    // Uploads one chunk; network errors, 5xx and 429 are retried with backoff, other errors are final.
    async function uploadChunk(formData, signal) {
        for (let attempt = 1; ; attempt++) {
            let response;
            try {
                response = await authFetch('/upload/chunk', { method: 'POST', body: formData, signal });
            } catch (error) {
                if (signal.aborted || error.message === 'Session expired' || attempt >= CHUNK_ATTEMPTS) throw error;
                await sleep(1000 * 2 ** (attempt - 1));
                continue;
            }
            if (response.ok) return;
            const retryable = response.status >= 500 || response.status === 429;
            if (!retryable || attempt >= CHUNK_ATTEMPTS) {
                throw new Error(await errorDetail(response, `Chunk upload failed (HTTP ${response.status})`));
            }
            await sleep(1000 * 2 ** (attempt - 1));
        }
    }

    async function uploadFileInChunks(entry) {
        const { file, taskType, options, row, controller } = entry;
        const uploadId = row.id;
        const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
        const statusCell = row.querySelector('.status-cell-value');
        statusCell.innerHTML = `<span class="job-status-badge status-uploading">uploading</span>${progressBarHtml(0)}`;
        const progressBar = statusCell.querySelector('.progress-bar');

        try {
            for (let chunkNumber = 0; chunkNumber < totalChunks; chunkNumber++) {
                const start = chunkNumber * CHUNK_SIZE;
                const formData = new FormData();
                formData.append('chunk', file.slice(start, Math.min(start + CHUNK_SIZE, file.size)), file.name);
                formData.append('upload_id', uploadId);
                formData.append('chunk_number', chunkNumber);
                await uploadChunk(formData, controller.signal);
                progressBar.style.width = `${Math.round(((chunkNumber + 1) / totalChunks) * 100)}%`;
            }
        } catch (error) {
            if (controller.signal.aborted) return;
            console.error(`Error uploading ${file.name}:`, error);
            if (error.message !== 'Session expired') showUploadFailure(row, 'Upload Failed', error.message);
            return;
        }

        // Finalize the upload. Not retried: a repeated finalize can't tell whether the first one created the job.
        if (controller.signal.aborted) return;
        try {
            const finalizePayload = { upload_id: uploadId, original_filename: file.name, total_chunks: totalChunks, task_type: taskType, ...options };
            const finalizeResponse = await authFetch('/upload/finalize', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(finalizePayload),
            });
            if (!finalizeResponse.ok) throw new Error(await errorDetail(finalizeResponse, 'Finalization failed'));
            const job = await finalizeResponse.json();
            row.remove();
            renderJobRow(job);
            handleSelectionChange();
            requestPoll();
        } catch (error) {
            console.error('Error finalizing upload:', error);
            if (error.message !== 'Session expired') showUploadFailure(row, 'Finalization Failed', error.message);
        }
    }

    function pumpUploadQueue() {
        while (activeUploads < MAX_PARALLEL_UPLOADS && uploadQueue.length > 0) {
            const entry = uploadQueue.shift();
            activeUploads++;
            uploadFileInChunks(entry).finally(() => {
                activeUploads--;
                uploadsById.delete(entry.row.id);
                pumpUploadQueue();
            });
        }
    }

    // Checks and queues a file for upload; returns false if it was rejected right away.
    function queueUpload(file, taskType, options) {
        const problem = uploadProblem(file, taskType);
        const row = createUploadRow(file, taskType, problem ? 'failed' : 'queued');
        if (problem) {
            showUploadFailure(row, 'Not Uploaded', problem);
            return false;
        }
        const entry = { file, taskType, options: { ...options }, row, controller: new AbortController() };
        uploadsById.set(row.id, entry);
        uploadQueue.push(entry);
        pumpUploadQueue();
        return true;
    }

    function cancelUpload(uploadId) {
        const entry = uploadsById.get(uploadId);
        if (!entry) return;
        const queuedAt = uploadQueue.indexOf(entry);
        if (queuedAt >= 0) {
            uploadQueue.splice(queuedAt, 1);
            uploadsById.delete(uploadId);
        }
        entry.controller.abort();
        entry.row.remove();
    }

    // Closing the tab would silently drop uploads that are still running.
    window.addEventListener('beforeunload', event => {
        if (activeUploads > 0 || uploadQueue.length > 0) {
            event.preventDefault();
            event.returnValue = '';
        }
    });

    function taskOptionsFromMainForm(taskType) {
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
        return options;
    }

    function handleTaskRequest(taskType) {
        if (mainFileInput.files.length === 0) return alert('Please choose one or more files first.');
        const options = taskOptionsFromMainForm(taskType);
        if (!options) return;
        // Uploads run in the background queue, so more files can be chosen right away.
        Array.from(mainFileInput.files).forEach(file => queueUpload(file, taskType, options));
        mainFileInput.value = '';
        updateFileName(mainFileInput, mainFileName);
    }

    function setupDragAndDropListeners() {
        let dragCounter = 0;
        window.addEventListener('dragenter', e => { e.preventDefault(); dragCounter++; document.body.classList.add('dragging'); });
        window.addEventListener('dragleave', e => { e.preventDefault(); dragCounter = Math.max(0, dragCounter - 1); if (dragCounter === 0) document.body.classList.remove('dragging'); });
        window.addEventListener('dragover', e => e.preventDefault());
        window.addEventListener('drop', e => {
            e.preventDefault();
            dragCounter = 0;
            document.body.classList.remove('dragging');
            if (e.target === dragOverlay || dragOverlay.contains(e.target)) {
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    stagedFiles = Array.from(e.dataTransfer.files);
                    showActionDialog();
                }
            }
        });
    }

    // --- Output formats ---
    function toolFormatGroup(toolKey, tool) {
        return {
            label: tool.name,
            choices: Object.keys(tool.formats || {}).map(formatKey => ({
                value: `${toolKey}_${formatKey}`,
                label: `${tool.name} - ${tool.formats[formatKey]}`
            }))
        };
    }

    // Grouped choices for every configured output format (used when no single input type is known).
    function allFormatChoices() {
        const tools = window.APP_CONFIG.conversionTools || {};
        return Object.keys(tools).map(toolKey => toolFormatGroup(toolKey, tools[toolKey]));
    }

    // Output formats of the tools that accept every selected file type. Computed in the browser
    // from the embedded tool list, so it needs no request. ZIP files are converted file by file,
    // so they don't restrict the list.
    function formatChoicesForFiles(files) {
        const extensions = [...new Set(Array.from(files || []).map(file => getFileExtension(file.name)))].filter(ext => ext !== '.zip');
        if (extensions.length === 0) return allFormatChoices();
        const tools = window.APP_CONFIG.conversionTools || {};
        const groups = Object.keys(tools)
            .filter(toolKey => extensions.every(ext => (tools[toolKey].supported_input || []).includes(ext)))
            .map(toolKey => toolFormatGroup(toolKey, tools[toolKey]));
        if (groups.length > 0) return groups;
        const label = files.length === 1 ? `No conversion available for ${extensions[0] || 'this file type'}` : 'No output format supports all selected files';
        return [{ value: '', label, disabled: true }];
    }

    function setFormatChoices(choices, files) {
        if (!choices) return;
        choices.clearStore();
        choices.setChoices(formatChoicesForFiles(files), 'value', 'label', true);
    }

    function showActionDialog() {
        dialogFileCount.textContent = stagedFiles.length;

        if (dialogConversionChoices) dialogConversionChoices.destroy();
        dialogConversionChoices = new Choices(dialogOutputFormatSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a format...' });
        setFormatChoices(dialogConversionChoices, stagedFiles);

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
            options.model_size = transcriptionChoices ? transcriptionChoices.getValue(true) : mainModelSizeSelect.value;
            options.generate_timestamps = document.getElementById('dialog-timestamps-checkbox').checked;
        } else if (action === 'tts') {
            const selectedModel = dialogTtsChoices.getValue(true);
            if (!selectedModel) return alert('Please select a voice model.');
            options.model_name = selectedModel;
        } else if (action === 'ocr') {
            options.ocr_language = selectedOcrLanguage();
        }
        stagedFiles.forEach(file => queueUpload(file, action, options));
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

    function setTtsPlaceholder(label) {
        ttsModelsCache = [{ value: '', label, disabled: true }];
        if (ttsChoices) ttsChoices.setChoices(ttsModelsCache, 'value', 'label', true);
    }

    async function loadTtsModels() {
        try {
            const response = await authFetch('/api/v1/tts-voices');
            if (response.status === 501) return setTtsPlaceholder('Text-to-speech is not available on this server');
            if (!response.ok) throw new Error(await errorDetail(response));
            const voicesData = await response.json();
            const voicesArray = [];
            if (Array.isArray(voicesData)) {
                voicesData.forEach(v => {
                    const id = v.id || v.voice_id || v.name;
                    if (id) voicesArray.push({ id, name: v.name || id, lang: (v.language && v.language.name) || v.locale || id.split(/[_-]/)[0] });
                });
            }
            if (voicesArray.length === 0) return setTtsPlaceholder('No voices found');
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
            if (error.message !== 'Session expired') setTtsPlaceholder('Error loading voices');
        }
    }

    function initializeSelectors() {
        conversionChoices = new Choices(mainOutputFormatSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select a format...' });
        conversionChoices.setChoices(allFormatChoices(), 'value', 'label', true);

        transcriptionChoices = new Choices(mainModelSizeSelect, { searchEnabled: false, shouldSort: false, itemSelectText: '' });

        ttsChoices = new Choices(mainTtsModelSelect, { searchEnabled: true, itemSelectText: 'Select', shouldSort: false, placeholder: true, placeholderValue: 'Select voice...' });
        loadTtsModels();

        if (mainOcrLanguageSelect) {
            ocrLanguageChoices = new Choices(mainOcrLanguageSelect, { removeItemButton: true, searchEnabled: true, itemSelectText: '', shouldSort: true, placeholder: true, placeholderValue: 'Server default' });
            loadOcrLanguages();
        }
    }

    function updateFileName(input, nameDisplay) {
        const numFiles = input.files.length;
        nameDisplay.textContent = numFiles === 1 ? input.files[0].name : (numFiles > 1 ? `${numFiles} files selected` : 'No files chosen');
        nameDisplay.title = numFiles > 1 ? Array.from(input.files).map(f => f.name).join(', ') : nameDisplay.textContent;
        // Only the formats that work for the chosen file(s); all formats when none is chosen.
        setFormatChoices(conversionChoices, input.files);
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

    // --- Job actions ---
    async function handleCancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) return;
        try {
            const response = await authFetch(`/job/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
            if (!response.ok) throw new Error(await errorDetail(response, 'Failed to cancel job.'));
            const result = await response.json();
            if (result.job) renderJobRow(result.job); // shows "cancelled" right away
            handleSelectionChange();
            requestPoll();
        } catch (error) {
            if (error.message !== 'Session expired') alert(`Error: ${error.message}`);
        }
    }

    function handleSelectionChange() {
        const checkboxes = jobListBody.querySelectorAll('.job-checkbox');
        const selectedCheckboxes = jobListBody.querySelectorAll('.job-checkbox:checked');
        downloadSelectedBtn.disabled = jobListBody.querySelectorAll('.job-checkbox[data-downloadable]:checked').length === 0;
        deleteSelectedBtn.disabled = selectedCheckboxes.length === 0;
        selectAllJobsCheckbox.checked = checkboxes.length > 0 && selectedCheckboxes.length === checkboxes.length;
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

    // The ZIP is requested with a form post into a hidden frame, so the browser streams it straight
    // to disk with its own progress display (fetch + blob held the whole archive in memory first).
    function downloadFrame() {
        let frame = document.getElementById('download-frame');
        if (frame) return frame;
        frame = document.createElement('iframe');
        frame.id = 'download-frame';
        frame.name = 'download-frame';
        frame.hidden = true;
        document.body.appendChild(frame);
        // Downloads don't load anything into the frame; an error response does.
        frame.addEventListener('load', () => {
            let text = '';
            try { text = (frame.contentDocument && frame.contentDocument.body && frame.contentDocument.body.textContent) || ''; } catch (e) { /* not readable */ }
            if (!text.trim()) return;
            let message = text.trim();
            try { message = JSON.parse(message).detail || message; } catch (e) { /* plain text */ }
            alert(`Could not download files: ${message}`);
        });
        return frame;
    }

    function handleBatchDownload() {
        const selectedIds = Array.from(jobListBody.querySelectorAll('.job-checkbox[data-downloadable]:checked')).map(cb => cb.value);
        if (selectedIds.length === 0) return;
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = apiUrl('/download/batch');
        form.target = downloadFrame().name;
        form.hidden = true;
        selectedIds.forEach(id => {
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = 'job_ids';
            input.value = id;
            form.appendChild(input);
        });
        document.body.appendChild(form);
        form.submit();
        form.remove();
        downloadSelectedBtn.disabled = true;
        downloadSelectedBtn.textContent = 'Preparing ZIP...';
        setTimeout(() => {
            downloadSelectedBtn.textContent = 'Download Selected as ZIP';
            handleSelectionChange();
        }, 2000);
    }

    async function loadInitialJobs(attempt = 1) {
        try {
            const response = await authFetch('/jobs');
            if (!response.ok) throw new Error('Failed to fetch jobs.');
            const jobs = await response.json();
            jobListBody.querySelectorAll('tr:not(.upload-row)').forEach(row => row.remove());
            renderedJobs.clear();
            pollErrors = 0;
            renderJobs(jobs);
            handleSelectionChange();
            requestPoll();
        } catch (error) {
            console.error("Couldn't load job history:", error);
            if (error.message === 'Session expired') return;
            pollErrors = Math.max(pollErrors, 1);
            updateConnectionStatus();
            if (!document.getElementById('jobs-load-error')) {
                jobListBody.insertAdjacentHTML('beforeend', '<tr id="jobs-load-error"><td colspan="7" style="text-align: center;">Could not load job history. Retrying…</td></tr>');
            }
            setTimeout(() => loadInitialJobs(attempt + 1), Math.min(POLL_ERROR_MAX_MS, 2000 * attempt));
        }
    }

    function toggleSubJobs(parentRow) {
        parentRow.classList.toggle('sub-jobs-visible');
        const visible = parentRow.classList.contains('sub-jobs-visible');
        const parentId = parentRow.id.replace('job-', '');
        jobListBody.querySelectorAll('tr').forEach(row => {
            if (row.dataset.parentId !== parentId) return;
            if (row.classList.contains('sub-job')) {
                row.classList.toggle('is-visible', visible);
                const button = row.querySelector('.details-button');
                if (button && !visible) button.textContent = 'i';
            } else if (!visible) {
                row.style.display = 'none'; // a sub-job's open details row closes with it
            }
        });
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
            const button = e.target.closest('button');
            if (button && button.classList.contains('cancel-button')) {
                e.preventDefault();
                if (button.dataset.jobId) handleCancelJob(button.dataset.jobId);
                else if (button.dataset.uploadId) cancelUpload(button.dataset.uploadId);
                else if (button.dataset.dismissUpload) document.getElementById(button.dataset.dismissUpload)?.remove();
                return;
            }

            // Details button (shown on small screens)
            if (button && button.classList.contains('details-button')) {
                const row = button.closest('tr');
                const detailsRow = document.getElementById(`${row.id}-details`);
                if (detailsRow) {
                    const isCurrentlyVisible = detailsRow.style.display !== 'none';
                    detailsRow.style.display = isCurrentlyVisible ? 'none' : 'table-row';
                    button.textContent = isCurrentlyVisible ? 'i' : '×';
                }
                return;
            }

            const parentRow = e.target.closest('tr.parent-job');
            if (parentRow && !e.target.closest('a, input')) toggleSubJobs(parentRow);
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
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && actionDialog.classList.contains('visible')) closeActionDialog();
        });

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
