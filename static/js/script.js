document.addEventListener('DOMContentLoaded', () => {
    // --- User Locale and Timezone Detection (Corrected Implementation) ---
    const USER_LOCALE = navigator.language || 'en-US'; // Fallback to en-US
    const USER_TIMEZONE = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const DATETIME_FORMAT_OPTIONS = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        timeZone: USER_TIMEZONE,
    };
    console.log(`Using locale: ${USER_LOCALE} and timezone: ${USER_TIMEZONE}`);

    // --- Element Selectors ---
    const jobListBody = document.getElementById('job-list-body');
    
    const pdfForm = document.getElementById('pdf-form');
    const pdfFileInput = document.getElementById('pdf-file-input');
    const pdfFileName = document.getElementById('pdf-file-name');
    
    const audioForm = document.getElementById('audio-form');
    const audioFileInput = document.getElementById('audio-file-input');
    const audioFileName = document.getElementById('audio-file-name');
    const modelSizeSelect = document.getElementById('model-size-select');
    
    const conversionForm = document.getElementById('conversion-form');
    const conversionFileInput = document.getElementById('conversion-file-input');
    const conversionFileName = document.getElementById('conversion-file-name');
    const outputFormatSelect = document.getElementById('output-format-select');

    // START: Drag and Drop additions
    const dragOverlay = document.getElementById('drag-overlay');
    const actionDialog = document.getElementById('action-dialog');
    const dialogFileCount = document.getElementById('dialog-file-count');
    // Dialog Views
    const dialogInitialView = document.getElementById('dialog-initial-actions');
    const dialogConvertView = document.getElementById('dialog-convert-view');
    // Dialog Buttons
    const dialogConvertBtn = document.getElementById('dialog-action-convert');
    const dialogOcrBtn = document.getElementById('dialog-action-ocr');
    const dialogTranscribeBtn = document.getElementById('dialog-action-transcribe');
    const dialogCancelBtn = document.getElementById('dialog-action-cancel');
    const dialogStartConversionBtn = document.getElementById('dialog-start-conversion');
    const dialogBackBtn = document.getElementById('dialog-back');
    // Dialog Select
    const dialogOutputFormatSelect = document.getElementById('dialog-output-format-select');
    // END: Drag and Drop additions

    let conversionChoices = null;
    let dialogConversionChoices = null; // For the dialog's format selector
    const activePolls = new Map();
    let stagedFiles = null; // To hold files from a drop event

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

    // --- Helper Functions ---
    function formatBytes(bytes, decimals = 1) {
        if (!+bytes) return '0 Bytes'; // Handles 0, null, undefined
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    // --- Core Job Submission Logic (Refactored for reuse) ---
    async function submitJob(endpoint, formData, originalFilename) {
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
                original_filename: originalFilename,
                input_filesize: formData.get('file').size,
                task_type: endpoint.includes('ocr') ? 'ocr' : (endpoint.includes('transcribe') ? 'transcription' : 'conversion'),
                created_at: new Date().toISOString() // Create preliminary UTC timestamp
            };
            renderJobRow(preliminaryJob);
            startPolling(result.job_id);
        } catch (error) {
            console.error('Error submitting job:', error);
            alert(`Submission failed for ${originalFilename}: ${error.message}`);
        }
    }

    // --- Original Form Submission Handler (Now uses submitJob) ---
    async function handleFormSubmit(event, endpoint, form) {
        event.preventDefault();
        const fileInput = form.querySelector('input[type="file"]');
        if (fileInput.files.length === 0) return;

        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = true;

        // Convert FileList to an array to loop through it
        const files = Array.from(fileInput.files);

        // Process each file as a separate job
        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);

            // Append other form data if it exists
            const outputFormat = form.querySelector('select[name="output_format"]');
            if (outputFormat) {
                formData.append('output_format', outputFormat.value);
            }
            const modelSize = form.querySelector('select[name="model_size"]');
            if (modelSize) {
                formData.append('model_size', modelSize.value);
            }

            // Await each job submission to process them sequentially
            await submitJob(endpoint, formData, file.name);
        }

        // Reset the form UI after all jobs have been submitted
        const fileNameDisplay = form.querySelector('.file-name');
        form.reset();
        if (fileNameDisplay) {
             fileNameDisplay.textContent = 'No file chosen';
             fileNameDisplay.title = 'No file chosen';
        }
        if (form.id === 'conversion-form' && conversionChoices) {
            conversionChoices.clearInput();
            conversionChoices.setValue([]);
        }
        submitButton.disabled = false;
    }

    // --- START: Drag and Drop Implementation ---
   function setupDragAndDropListeners() {
        let dragCounter = 0; // Counter to manage enter/leave events reliably

        window.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dragCounter++;
            document.body.classList.add('dragging');
        });

        window.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter === 0) {
                document.body.classList.remove('dragging');
            }
        });

        window.addEventListener('dragover', (e) => {
            e.preventDefault(); // This is necessary to allow a drop
        });

        window.addEventListener('drop', (e) => {
            e.preventDefault();
            dragCounter = 0; // Reset counter
            document.body.classList.remove('dragging');
            
            // Only handle the drop if it's on our designated overlay
            if (e.target === dragOverlay || dragOverlay.contains(e.target)) {
                const files = e.dataTransfer.files;
                if (files && files.length > 0) {
                    stagedFiles = files;
                    showActionDialog();
                }
            }
        });
    }

    function showActionDialog() {
        dialogFileCount.textContent = stagedFiles.length;

        // Clone options from main form's select to the dialog's select
        dialogOutputFormatSelect.innerHTML = outputFormatSelect.innerHTML;

        // Clean up previous Choices.js instance if it exists
        if (dialogConversionChoices) {
            dialogConversionChoices.destroy();
        }

        // Initialize a new Choices.js instance for the dialog
        dialogConversionChoices = new Choices(dialogOutputFormatSelect, {
            searchEnabled: true,
            itemSelectText: 'Select',
            shouldSort: false,
            placeholder: true,
            placeholderValue: 'Select a format...',
        });

        // Ensure the initial view is shown
        dialogInitialView.style.display = 'grid';
        dialogConvertView.style.display = 'none';
        actionDialog.classList.add('visible');
    }

    function closeActionDialog() {
        actionDialog.classList.remove('visible');
        stagedFiles = null;
        // Important: Destroy the Choices instance to prevent memory leaks
        if (dialogConversionChoices) {
            // Explicitly hide the dropdown before destroying
            dialogConversionChoices.hideDropdown(); 
            dialogConversionChoices.destroy();
            dialogConversionChoices = null;
        }
    }

    // --- Dialog Button and Action Listeners ---
    dialogConvertBtn.addEventListener('click', () => {
        // Switch to the conversion view
        dialogInitialView.style.display = 'none';
        dialogConvertView.style.display = 'block';
    });

    dialogBackBtn.addEventListener('click', () => {
        // Switch back to the initial view
        dialogInitialView.style.display = 'grid';
        dialogConvertView.style.display = 'none';
    });

    dialogStartConversionBtn.addEventListener('click', () => handleDialogAction('convert'));
    dialogOcrBtn.addEventListener('click', () => handleDialogAction('ocr'));
    dialogTranscribeBtn.addEventListener('click', () => handleDialogAction('transcribe'));
    dialogCancelBtn.addEventListener('click', closeActionDialog);


    function handleDialogAction(action) {
        if (!stagedFiles) return;

        let endpoint = '';
        const formDataArray = [];

        for (const file of stagedFiles) {
            const formData = new FormData();
            formData.append('file', file);
            
            if (action === 'convert') {
                const selectedFormat = dialogConversionChoices.getValue(true);
                if (!selectedFormat) {
                    alert('Please select a format to convert to.');
                    return;
                }
                formData.append('output_format', selectedFormat);
                endpoint = '/convert-file';
            } else if (action === 'ocr') {
                endpoint = '/ocr-pdf';
            } else if (action === 'transcribe') {
                formData.append('model_size', modelSizeSelect.value);
                endpoint = '/transcribe-audio';
            }
            formDataArray.push({ formData, name: file.name });
        }

        formDataArray.forEach(item => {
            submitJob(endpoint, item.formData, item.name);
        });

        closeActionDialog();
    }
    // --- END: Drag and Drop Implementation ---

    function initializeConversionSelector() {
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
                    label: `${tool.name} - ${formatKey.toUpperCase()} (${tool.formats[formatKey]})`
                });
            }
            choicesArray.push(group);
        }
        conversionChoices.setChoices(choicesArray, 'value', 'label', true);
    }
    
    function updateFileName(input, nameDisplay) {
        const numFiles = input.files.length;
        let displayText = 'No file chosen';
        let displayTitle = 'No file chosen';

        if (numFiles === 1) {
            displayText = input.files[0].name;
            displayTitle = input.files[0].name;
        } else if (numFiles > 1) {
            displayText = `${numFiles} files selected`;
            // Create a title attribute to show all filenames on hover
            displayTitle = Array.from(input.files).map(file => file.name).join(', ');
        }
        nameDisplay.textContent = displayText;
        nameDisplay.title = displayTitle;
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
            jobListBody.innerHTML = '<tr><td colspan="6" style="text-align: center;">Could not load job history.</td></tr>';
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

        // --- CORRECTED DATE FORMATTING ---
        // Takes the UTC string from the server (or the preliminary job)
        // and formats it using the user's detected locale and timezone.
        const submittedDate = new Date(job.created_at);
        const formattedDate = submittedDate.toLocaleString(USER_LOCALE, DATETIME_FORMAT_OPTIONS);

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

        // --- File Size Logic ---
        let fileSizeHtml = '<span>-</span>';
        if (job.input_filesize) {
            let sizeString = formatBytes(job.input_filesize);
            if (job.status === 'completed' && job.output_filesize) {
                sizeString += ` → ${formatBytes(job.output_filesize)}`;
            }
            fileSizeHtml = `<span class="cell-value">${sizeString}</span>`;
        }

        const escapedFilename = job.original_filename ? job.original_filename.replace(/</g, "&lt;").replace(/>/g, "&gt;") : "No filename";
        
        row.innerHTML = `
            <td data-label="File"><span class="cell-value" title="${escapedFilename}">${escapedFilename}</span></td>
            <td data-label="File Size">${fileSizeHtml}</td>
            <td data-label="Task"><span class="cell-value">${taskTypeLabel}</span></td>
            <td data-label="Submitted"><span class="cell-value">${formattedDate}</span></td>
            <td data-label="Status"><span class="cell-value">${statusHtml}</span></td>
            <td data-label="Action" class="action-col"><span class="cell-value">${actionHtml}</span></td>
        `;
    }

    // --- Initial Load ---
    initializeConversionSelector();
    loadInitialJobs();
    setupDragAndDropListeners();
});