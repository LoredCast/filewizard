// static/js/settings.js
document.addEventListener('DOMContentLoaded', () => {
    const settingsForm = document.getElementById('settings-form');
    const saveStatus = document.getElementById('save-status');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const deleteFilesBtn = document.getElementById('delete-files-btn');

    // --- Save Settings ---
    settingsForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        saveStatus.textContent = 'Saving...';
        saveStatus.classList.remove('success', 'error');

        const formData = new FormData(settingsForm);
        const settingsObject = {};

        // Convert FormData to a nested object
        formData.forEach((value, key) => {
            // Handle checkboxes that might not be submitted if unchecked
            if (key.includes('ocr_settings')) {
                 const checkbox = document.querySelector(`[name="${key}"]`);
                 if (checkbox && checkbox.type === 'checkbox') {
                    value = checkbox.checked;
                 }
            }

            const keys = key.split('.');
            let current = settingsObject;
            keys.forEach((k, index) => {
                if (index === keys.length - 1) {
                    current[k] = value;
                } else {
                    current[k] = current[k] || {};
                    current = current[k];
                }
            });
        });
        
        // Ensure unchecked OCR boxes are sent as false
        const ocrCheckboxes = settingsForm.querySelectorAll('input[type="checkbox"][name^="ocr_settings"]');
        ocrCheckboxes.forEach(cb => {
            const keys = cb.name.split('.');
            if (!formData.has(cb.name)) {
                 // this is a bit of a hack but gets the job done for this specific form
                 settingsObject[keys[0]][keys[1]][keys[2]] = false;
            }
        });


        try {
            const response = await fetch('/settings/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settingsObject)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save settings.');
            }

            saveStatus.textContent = 'Settings saved successfully!';
            saveStatus.classList.add('success');

        } catch (error) {
            saveStatus.textContent = `Error: ${error.message}`;
            saveStatus.classList.add('error');
            console.error('Error saving settings:', error);
        } finally {
            setTimeout(() => {
                saveStatus.textContent = '';
                saveStatus.classList.remove('success', 'error');
            }, 5000);
        }
    });

    // --- Clear History ---
    clearHistoryBtn.addEventListener('click', async () => {
        if (!confirm('ARE YOU SURE?\n\nThis will permanently delete all job history records from the database.')) {
            return;
        }
        try {
            const response = await fetch('/settings/clear-history', { method: 'POST' });
            if (!response.ok) throw new Error('Server responded with an error.');
            const result = await response.json();
            alert(`Success: ${result.deleted_count} job records have been deleted.`);
        } catch (error) {
            alert('An error occurred while clearing history.');
            console.error(error);
        }
    });

    // --- Delete Files ---
    deleteFilesBtn.addEventListener('click', async () => {
        if (!confirm('ARE YOU SURE?\n\nThis will permanently delete all files in the "processed" folder.')) {
            return;
        }
        try {
            const response = await fetch('/settings/delete-files', { method: 'POST' });
            if (!response.ok) throw new Error('Server responded with an error.');
            const result = await response.json();
            alert(`Success: ${result.deleted_count} files have been deleted.`);
        } catch (error) {
            alert('An error occurred while deleting files.');
            console.error(error);
        }
    });
});