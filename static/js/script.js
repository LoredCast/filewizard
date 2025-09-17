// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selectors ---
    const jobListBody = document.getElementById('job-list-body');
    
    const pdfForm = document.getElementById('pdf-form');
    const pdfFileInput = document.getElementById('pdf-file-input');
    const pdfFileName = document.getElementById('pdf-file-name');
    
    const audioForm = document.getElementById('audio-form');
    const audioFileInput = document.getElementById('audio-file-input');
    const audioFileName = document.getElementById('audio-file-name');
    
    const conversionForm = document.getElementById('conversion-form');
    const conversionFileInput = document.getElementById('conversion-file-input');
    const conversionFileName = document.getElementById('conversion-file-name');
    const outputFormatSelect = document.getElementById('output-format-select');

    // MODIFICATION: Store the Choices.js instance in a variable
    let conversionChoices = null;

    const activePolls = new Map();

    // --- Main Event Listeners ---
    pdfFileInput.addEventListener('change', () => updateFileName(pdfFileInput, pdfFileName));
    audioFileInput.addEventListener('change', () => updateFileName(audioFileInput, audioFileName));
    conversionFileInput.addEventListener('change', () => updateFileName(conversionFileInput, conversionFileName));

    pdfForm.addEventListener('submit', (e) => handleFormSubmit(e, '/ocr-pdf', pdfForm));
    audioForm.addEventListener('submit', (e) => handleFormSubmit(e, '/transcribe-audio', audioForm));
    conversionForm.addEventListener('submit', (e) => handleFormSubmit(e, '/convert-file', conversionForm));

    jobListBody.addEventListener('click', (event) => {
        if (event.target.classList.contains('cancel-button')) {
            const jobId = event.target.dataset.jobId;
            handleCancelJob(jobId);
        }
    });
    
    function initializeConversionSelector() {
        // MODIFICATION: Destroy the old instance if it exists before creating a new one
        if (conversionChoices) {
            conversionChoices.destroy();
        }

        conversionChoices = new Choices(outputFormatSelect, {
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
            const group = {
                label: tool.name,
                id: toolKey,
                disabled: false,
                choices: []
            };
            for (const formatKey in tool.formats) {
                group.choices.push({
                    value: `${toolKey}_${formatKey}`,
                    label: `${formatKey.toUpperCase()} - ${tool.formats[formatKey]}`
                });
            }
            choicesArray.push(group);
        }
        conversionChoices.setChoices(choicesArray, 'value', 'label', true);
    }
    
    // --- Helper Functions ---
    function updateFileName(input, nameDisplay) {
        const fileName = input.files.length > 0 ? input.files[0].name : 'No file chosen';
        nameDisplay.textContent = fileName;
        nameDisplay.title = fileName;
    }

    async function handleFormSubmit(event, endpoint, form) {
        event.preventDefault();
        const fileInput = form.querySelector('input[type="file"]');
        const fileNameDisplay = form.querySelector('.file-name');
        if (!fileInput.files[0]) return;

        const formData = new FormData(form);
        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = true;

        try {
            const response = await fetch(endpoint, { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }
            const result = await response.json();
            const preliminaryJob = {
                id: result.job_id,
                status: 'pending',
                progress: 0,
                original_filename: fileInput.files[0].name,
                task_type: endpoint.includes('ocr') ? 'ocr' : (endpoint.includes('transcribe') ? 'transcription' : 'conversion'),
                created_at: new Date().toISOString()
            };
            renderJobRow(preliminaryJob);
            startPolling(result.job_id);
        } catch (error) {
            console.error('Error submitting job:', error);
            alert(`Submission failed: ${error.message}`);
        } finally {
            form.reset();
            if (fileNameDisplay) fileNameDisplay.textContent = 'No file chosen';
            
            // MODIFICATION: Use the stored instance to correctly reset the dropdown
            // without causing an error.
            if (form.id === 'conversion-form' && conversionChoices) {
                 conversionChoices.clearInput();
                 conversionChoices.setValue([]); // Clears the selected value
            }
            
            submitButton.disabled = false;
        }
    }
    
    async function handleCancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) return;
        try {
            const response = await fetch(`/job/${jobId}/cancel`, { method: 'POST' });
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
            alert(`Error: ${error.message}`);
        }
    }

    async function loadInitialJobs() {
        try {
            const response = await fetch('/jobs');
            if (!response.ok) throw new Error('Failed to fetch jobs.');
            const jobs = await response.json();
            jobListBody.innerHTML = '';
            for (const job of jobs.reverse()) {
                renderJobRow(job);
                if (['pending', 'processing'].includes(job.status)) {
                    startPolling(job.id);
                }
            }
        } catch (error) {
            console.error("Couldn't load job history:", error);
            jobListBody.innerHTML = '<tr><td colspan="5" style="text-align: center;">Could not load job history.</td></tr>';
        }
    }

    function startPolling(jobId) {
        if (activePolls.has(jobId)) return;
        const intervalId = setInterval(async () => {
            try {
                const response = await fetch(`/job/${jobId}`);
                if (!response.ok) {
                    if (response.status === 404) stopPolling(jobId);
                    return;
                }
                const job = await response.json();
                renderJobRow(job);
                if (['completed', 'failed', 'cancelled'].includes(job.status)) {
                    stopPolling(jobId);
                }
            } catch (error) {
                console.error(`Error polling for job ${jobId}:`, error);
                stopPolling(jobId);
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
        let row = document.getElementById(`job-${job.id}`);
        if (!row) {
            row = document.createElement('tr');
            row.id = `job-${job.id}`;
            jobListBody.prepend(row);
        }
        
        let taskTypeLabel = 'Unknown';
        if (job.task_type === 'ocr' || job.task_type === 'ocr-image') {
            taskTypeLabel = 'OCR';
        } else if (job.task_type === 'transcription') {
            taskTypeLabel = 'Transcription';
        } else if (job.task_type === 'conversion' && job.processed_filepath) {
            const extension = job.processed_filepath.split('.').pop();
            taskTypeLabel = `Convert to ${extension.toUpperCase()}`;
        } else if (job.task_type === 'conversion') {
            taskTypeLabel = 'Conversion';
        }

        const formattedDate = new Date(job.created_at).toLocaleString();
        let statusHtml = `<span class="job-status-badge status-${job.status}">${job.status}</span>`;
        if (job.status === 'processing') {
            const progressClass = (job.task_type === 'transcription' && job.progress > 0) ? '' : 'indeterminate';
            const progressWidth = job.task_type === 'transcription' ? job.progress : 100;
            statusHtml += `<div class="progress-bar-container"><div class="progress-bar ${progressClass}" style="width: ${progressWidth}%"></div></div>`;
        }

        let actionHtml = `<span>-</span>`;
        if (job.status === 'pending' || job.status === 'processing') {
            actionHtml = `<button class="cancel-button" data-job-id="${job.id}">Cancel</button>`;
        } else if (job.status === 'completed' && job.processed_filepath) {
            const downloadFilename = job.processed_filepath.split(/[\\/]/).pop();
            actionHtml = `<a href="/download/${downloadFilename}" class="download-button" download>Download</a>`;
        } else if (job.status === 'failed') {
            const errorTitle = job.error_message ? ` title="${job.error_message.replace(/"/g, '&quot;')}"` : '';
            actionHtml = `<span class="error-text"${errorTitle}>Failed</span>`;
        }

        const escapedFilename = job.original_filename ? job.original_filename.replace(/</g, "&lt;").replace(/>/g, "&gt;") : "No filename";
        row.innerHTML = `
            <td data-label="File"><span class="cell-value" title="${escapedFilename}">${escapedFilename}</span></td>
            <td data-label="Task"><span class="cell-value">${taskTypeLabel}</span></td>
            <td data-label="Submitted"><span class="cell-value">${formattedDate}</span></td>
            <td data-label="Status"><span class="cell-value">${statusHtml}</span></td>
            <td data-label="Action" class="action-col"><span class="cell-value">${actionHtml}</span></td>
        `;
    }

    // --- Initial Load ---
    initializeConversionSelector();
    loadInitialJobs();
});