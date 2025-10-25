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

        const settingsObject = {};
        const elements = Array.from(settingsForm.elements);

        for (const el of elements) {
            if (!el.name || el.type === 'submit') continue; // Skip elements without a name and submit buttons

            let value;
            const keys = el.name.split('.');

            // Determine value based on element type
            if (el.type === 'checkbox') {
                value = el.checked;
            } else if (el.name.endsWith('.command_template')) {
                value = el.value;
            } else if (el.name.endsWith('.supported_input') || el.name === 'app_settings.allowed_all_extensions' || el.name === 'auth_settings.admin_users' || el.name === 'webhook_settings.allowed_callback_urls') {
                // Convert comma-separated text into an array of strings
                value = el.value.split(',')
                    .map(item => item.trim())
                    .filter(item => item); // Remove empty strings from the list
            } else if (el.name.endsWith('.formats')) {
                // Parse key:value pairs from textarea
                const lines = el.value.split('\n');
                const formatsObj = {};
                for (const line of lines) {
                    const parts = line.split(':');
                    if (parts.length >= 2) {
                        const key = parts[0].trim();
                        const value = parts.slice(1).join(':').trim();
                        if (key) {
                            formatsObj[key] = value;
                        }
                    }
                }
                value = formatsObj;
            } else if (el.tagName === 'TEXTAREA' && !el.name.endsWith('.command_template')) {
                // For any other text area, assume it's a comma separated list
                value = el.value.split(',')
                    .map(item => item.trim())
                    .filter(item => item);
            } else if (el.type === 'number') {
                value = parseFloat(el.value);
                if (isNaN(value)) {
                    value = null; // Represent empty number fields as null
                }
            } else {
                value = el.value;
            }

            // Build nested object from dot-notation name
            let current = settingsObject;
            keys.forEach((k, index) => {
                if (index === keys.length - 1) {
                    current[k] = value;
                } else {
                    if (!current[k]) {
                        current[k] = {};
                    }
                    current = current[k];
                }
            });
        }

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
