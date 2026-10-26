// Main JavaScript functionality for Vehicle Eligibility Prediction System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // File upload validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Check file size (16MB limit)
                const maxSize = 16 * 1024 * 1024; // 16MB in bytes
                if (file.size > maxSize) {
                    alert('File size exceeds 16MB limit. Please choose a smaller file.');
                    e.target.value = '';
                    return;
                }

                // Check file type
                const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];
                if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
                    alert('Please select a CSV file.');
                    e.target.value = '';
                    return;
                }

                // Update UI to show selected file
                const fileName = file.name;
                const fileInfo = document.createElement('small');
                fileInfo.className = 'text-success d-block mt-1';
                fileInfo.innerHTML = `<i class="fas fa-check me-1"></i>Selected: ${fileName}`;
                
                // Remove any existing file info
                const existingInfo = e.target.parentNode.querySelector('.text-success');
                if (existingInfo) {
                    existingInfo.remove();
                }
                
                e.target.parentNode.appendChild(fileInfo);
            }
        });
    });

    // Form validation for single prediction
    const singlePredictionForm = document.querySelector('form[action*="predict_single"]');
    if (singlePredictionForm) {
        singlePredictionForm.addEventListener('submit', function(e) {
            const inputs = this.querySelectorAll('input[type="number"]');
            let isValid = true;
            
            inputs.forEach(input => {
                const value = parseFloat(input.value);
                
                // Remove any existing error styling
                input.classList.remove('is-invalid');
                
                // Basic validation
                if (isNaN(value)) {
                    input.classList.add('is-invalid');
                    isValid = false;
                }
                
                // Specific field validations based on typical ranges
                const fieldName = input.name.toLowerCase();
                if (fieldName.includes('ram') && (value < 0 || value > 200)) {
                    showFieldWarning(input, 'RAM value seems unusual (expected: 0-200 GB)');
                } else if (fieldName.includes('storage') && (value < 0 || value > 1000)) {
                    showFieldWarning(input, 'Storage value seems unusual (expected: 0-1000 GB)');
                } else if (fieldName.includes('trust') && (value < 0 || value > 100)) {
                    showFieldWarning(input, 'Trust factor should be between 0-100');
                } else if (fieldName.includes('distance') && (value < 0 || value > 1000)) {
                    showFieldWarning(input, 'Distance value seems unusual (expected: 0-1000 km)');
                } else if (fieldName.includes('transmission') && (value < 0 || value > 100)) {
                    showFieldWarning(input, 'Transmission rate should be between 0-100%');
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                showAlert('Please correct the highlighted fields before submitting.', 'danger');
            } else {
                // Show loading state
                const submitBtn = this.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Predicting...';
                }
            }
        });
    }

    // Model selection validation
    const modelTrainingForm = document.querySelector('form[action*="train_models"]');
    if (modelTrainingForm) {
        modelTrainingForm.addEventListener('submit', function(e) {
            const checkboxes = this.querySelectorAll('input[type="checkbox"]:checked');
            
            if (checkboxes.length === 0) {
                e.preventDefault();
                showAlert('Please select at least one model to train.', 'warning');
                return;
            }
            
            // Show loading state
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Training Models...';
            }
        });
    }

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-info):not(.alert-light)');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert && alert.classList.contains('fade')) {
                alert.classList.remove('show');
                setTimeout(() => alert.remove(), 300);
            }
        }, 5000);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Utility functions
function showFieldWarning(input, message) {
    input.classList.add('is-warning');
    
    // Remove any existing warning
    const existingWarning = input.parentNode.querySelector('.text-warning');
    if (existingWarning) {
        existingWarning.remove();
    }
    
    // Add warning message
    const warning = document.createElement('small');
    warning.className = 'text-warning d-block mt-1';
    warning.innerHTML = `<i class="fas fa-exclamation-triangle me-1"></i>${message}`;
    input.parentNode.appendChild(warning);
    
    // Remove warning on input change
    input.addEventListener('input', function() {
        this.classList.remove('is-warning');
        const warningEl = this.parentNode.querySelector('.text-warning');
        if (warningEl) {
            warningEl.remove();
        }
    });
}

function showAlert(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show mt-3`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of main content
    const mainContent = document.querySelector('main.container');
    if (mainContent) {
        mainContent.insertBefore(alertContainer, mainContent.firstChild);
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertContainer.classList.remove('show');
        setTimeout(() => alertContainer.remove(), 300);
    }, 5000);
}

// Performance metrics chart (if Chart.js is needed later)
function createPerformanceChart(data, containerId) {
    // This function can be extended to create interactive charts
    // using Chart.js or D3.js if needed in the future
    console.log('Performance data ready for visualization:', data);
}

// Export functions for potential use in other scripts
window.MLApp = {
    showAlert,
    showFieldWarning,
    createPerformanceChart
};
