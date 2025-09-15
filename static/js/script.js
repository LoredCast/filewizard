// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    const jobListBody = document.getElementById('job-list-body');
    const pdfForm = document.getElementById('pdf-form');
    const pdfFileInput = document.getElementById('pdf-file-input');
    const pdfFileName = document.getElementById('pdf-file-name');
    const audioForm = document.getElementById('audio-form');
    const audioFileInput = document.getElementById('audio-file-input');
    const audioFileName = document.getElementById('audio-file-name');
    
    const activePolls = new Map();

    // --- Main Event Listeners ---
    pdfFileInput.addEventListener('change', () => updateFileName(pdfFileInput, pdfFileName));
    audioFileInput.addEventListener('change', () => updateFileName(audioFileInput, audioFileName));
    pdfForm.addEventListener('submit', (e) => handleFormSubmit(e, '/ocr-pdf', pdfForm, pdfFileInput, pdfFileName));
    audioForm.addEventListener('submit', (e) => handleFormSubmit(e, '/transcribe-audio', audioForm, audioFileInput, audioFileName));

    jobListBody.addEventListener('click', (event) => {
        if (event.target.classList.contains('cancel-button')) {
            const jobId = event.target.dataset.jobId;
            handleCancelJob(jobId);
        }
    });
    
    function updateFileName(input, nameDisplay) {
        nameDisplay.textContent = input.files.length > 0 ? input.files[0].name : 'No file chosen';
        nameDisplay.title = nameDisplay.textContent; // Add a tooltip for the full name
    }

    async function handleFormSubmit(event, endpoint, form, fileInput, fileNameDisplay) {
        event.preventDefault();
        if (!fileInput.files[0]) return;

        // MODIFICATION: Use new FormData(form) to capture all form fields,
        // including the new model size dropdown for the audio form.
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
            
            // To provide immediate feedback, create a placeholder job object
            const preliminaryJob = {
                id: result.job_id,
                status: 'pending',
                progress: 0,
                original_filename: fileInput.files[0].name,
                task_type: endpoint.includes('ocr') ? 'ocr' : 'transcription',
                created_at: new Date().toISOString(),
                processed_filepath: null,
                error_message: null
            };
            renderJobRow(preliminaryJob); // Render immediately
            startPolling(result.job_id); // Start polling for updates
        } catch (error) {
            console.error('Error submitting job:', error);
            alert(`Submission failed: ${error.message}`);
        } finally {
            form.reset();
            fileNameDisplay.textContent = 'No file chosen';
            fileNameDisplay.title = '';
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
            // The polling mechanism will update the UI to "cancelled" automatically.
            // We can stop polling immediately to be more efficient.
            stopPolling(jobId);
            // Optionally, force an immediate UI update
            const row = document.getElementById(`job-${jobId}`);
            if (row) {
                const statusCell = row.querySelector('td[data-label="Status"] .cell-value');
                const actionCell = row.querySelector('td[data-label="Action"] .cell-value');
                if (statusCell) {
                    statusCell.innerHTML = `<span class="job-status-badge status-cancelled">cancelled</span>`;
                }
                if (actionCell) {
                    actionCell.innerHTML = `<span>-</span>`;
                }
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
            jobListBody.innerHTML = ''; // Clear existing
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
                    if (response.status === 404) {
                        console.warn(`Job ${jobId} not found. Stopping poll.`);
                        stopPolling(jobId);
                    }
                    return;
                }
                const job = await response.json();
                renderJobRow(job);
                if (['completed', 'failed', 'cancelled'].includes(job.status)) {
                    stopPolling(jobId);
                }
            } catch (error) {
                console.error(`Error polling for job ${jobId}:`, error);
                stopPolling(jobId); // Stop polling on network or other errors
            }
        }, 2500); // Poll every 2.5 seconds
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

        const taskTypeLabel = job.task_type.includes('ocr') ? 'PDF/Image OCR' : 'Transcription';
        const formattedDate = new Date(job.created_at).toLocaleString();

        let statusHtml = `<span class="job-status-badge status-${job.status}">${job.status}</span>`;
        let actionHtml = `<span>-</span>`;

        if (job.status === 'processing') {
            // Show real progress for transcription, but an indeterminate one for OCR tasks
            const progressClass = job.task_type === 'transcription' ? '' : 'indeterminate';
            const progressWidth = job.task_type === 'transcription' ? job.progress : 100;
            const progressBarHtml = `<div class="progress-bar-container"><div class="progress-bar ${progressClass}" style="width: ${progressWidth}%"></div></div>`;
            statusHtml += progressBarHtml;
        }

        if (job.status === 'pending' || job.status === 'processing') {
            actionHtml = `<button class="cancel-button" data-job-id="${job.id}">Cancel</button>`;
        } else if (job.status === 'completed' && job.processed_filepath) {
            const downloadFilename = job.processed_filepath.split(/[\\/]/).pop();
            actionHtml = `<a href="/download/${downloadFilename}" class="download-button" download>Download</a>`;
        } else if (job.status === 'failed') {
            const errorTitle = job.error_message ? ` title="${job.error_message.replace(/"/g, '&quot;')}"` : '';
            actionHtml = `<span class="error-text"${errorTitle}>Failed</span>`;
        }

        // Use textContent for filename to prevent XSS and add a title for overflow
        const escapedFilename = job.original_filename.replace(/</g, "&lt;").replace(/>/g, "&gt;");

        row.innerHTML = `
            <td data-label="File"><span class="cell-value" title="${escapedFilename}">${escapedFilename}</span></td>
            <td data-label="Type"><span class="cell-value">${taskTypeLabel}</span></td>
            <td data-label="Submitted"><span class="cell-value">${formattedDate}</span></td>
            <td data-label="Status"><span class="cell-value">${statusHtml}</span></td>
            <td data-label="Action" class="action-col"><span class="cell-value">${actionHtml}</span></td>
        `;
    }

    // --- Initial Load ---
    loadInitialJobs();
});