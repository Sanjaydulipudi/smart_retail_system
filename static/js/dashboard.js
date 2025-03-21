// dashboard.js - Smart Retail System Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard components
    initializeDashboard();
    
    // Set up event listeners
    setupEventListeners();
});

/**
 * Initialize all dashboard components
 */
function initializeDashboard() {
    // Load summary stats
    loadSummaryStatistics();
    
    // Initialize charts if they exist on the page
    if (document.getElementById('salesTrendChart')) {
        initializeSalesTrendChart();
    }
    
    if (document.getElementById('categoryAnalysisChart')) {
        initializeCategoryAnalysisChart();
    }
    
    if (document.getElementById('salesPredictionChart')) {
        initializeSalesPredictionChart();
    }
    
    // Check for report images and load them
    loadReportImages();
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // Time period selector for charts
    const timePeriodSelector = document.getElementById('timePeriodSelector');
    if (timePeriodSelector) {
        timePeriodSelector.addEventListener('change', function() {
            updateChartsTimePeriod(this.value);
        });
    }
    
    // Category filter for category analysis
    const categoryFilter = document.getElementById('categoryFilter');
    if (categoryFilter) {
        categoryFilter.addEventListener('change', function() {
            updateCategoryChart(this.value);
        });
    }
    
    // Refresh data button
    const refreshButton = document.getElementById('refreshDataBtn');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            refreshDashboardData();
        });
    }
    
    // Toggle view buttons (if any)
    const viewToggleBtns = document.querySelectorAll('.view-toggle-btn');
    viewToggleBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const viewType = this.dataset.view;
            toggleView(viewType);
        });
    });
}

/**
 * Load summary statistics from the API
 */
function loadSummaryStatistics() {
    fetch('/api/data/summary')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateSummaryCards(data.data);
            } else {
                console.error('Error loading summary data:', data.error);
                showErrorMessage('Failed to load summary statistics');
            }
        })
        .catch(error => {
            console.error('API request failed:', error);
            showErrorMessage('Network error while loading statistics');
        });
}

/**
 * Update summary statistic cards with data
 */
function updateSummaryCards(summaryData) {
    // Format numbers and update DOM elements
    if (document.getElementById('totalSales')) {
        document.getElementById('totalSales').textContent = formatCurrency(summaryData.total_sales);
    }
    
    if (document.getElementById('avgDailySales')) {
        document.getElementById('avgDailySales').textContent = formatCurrency(summaryData.avg_daily_sales);
    }
    
    if (document.getElementById('salesGrowth')) {
        const growth = ((summaryData.max_sales / summaryData.min_sales - 1) * 100).toFixed(1);
        document.getElementById('salesGrowth').textContent = `${growth}%`;
        
        // Add color based on positive/negative growth
        if (parseFloat(growth) >= 0) {
            document.getElementById('salesGrowth').classList.add('text-success');
        } else {
            document.getElementById('salesGrowth').classList.add('text-danger');
        }
    }
    
    if (document.getElementById('recordCount')) {
        document.getElementById('recordCount').textContent = summaryData.record_count.toLocaleString();
    }
}

/**
 * Initialize sales trend chart
 */
function initializeSalesTrendChart() {
    fetch('/api/data/latest')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderSalesTrendChart(data.data);
            } else {
                console.error('Error loading latest data:', data.error);
                showErrorMessage('Failed to load sales trend data');
            }
        })
        .catch(error => {
            console.error('API request failed:', error);
            showErrorMessage('Network error while loading sales trend');
        });
}

/**
 * Render sales trend chart using Chart.js
 */
function renderSalesTrendChart(salesData) {
    const ctx = document.getElementById('salesTrendChart').getContext('2d');
    
    // Extract dates and sales values
    const dates = salesData.map(item => item.date);
    const sales = salesData.map(item => item.sales);
    
    // Create chart
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Sales',
                data: sales,
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                pointRadius: 2,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                },
                x: {
                    ticks: {
                        maxTicksLimit: 10
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return '$' + context.raw.toLocaleString();
                        }
                    }
                },
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Sales Trend'
                }
            }
        }
    });
}

/**
 * Initialize category analysis chart
 */
function initializeCategoryAnalysisChart() {
    // This would typically fetch category data from an API endpoint
    // For demonstration, we'll use sample data
    const categoryData = {
        labels: ['Electronics', 'Clothing', 'Groceries', 'Home Goods', 'Beauty'],
        datasets: [{
            label: 'Sales by Category',
            data: [35000, 25000, 18000, 12000, 10000],
            backgroundColor: [
                'rgba(255, 99, 132, 0.7)',
                'rgba(54, 162, 235, 0.7)',
                'rgba(255, 206, 86, 0.7)',
                'rgba(75, 192, 192, 0.7)',
                'rgba(153, 102, 255, 0.7)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)'
            ],
            borderWidth: 1
        }]
    };
    
    renderCategoryAnalysisChart(categoryData);
}

/**
 * Render category analysis chart
 */
function renderCategoryAnalysisChart(categoryData) {
    const ctx = document.getElementById('categoryAnalysisChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'pie',
        data: categoryData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: $${value.toLocaleString()} (${percentage}%)`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Sales by Category'
                }
            }
        }
    });
}

/**
 * Initialize sales prediction chart
 */
function initializeSalesPredictionChart() {
    // This might come from another API endpoint
    // For demonstration, we'll use sample data
    const predictionData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [
            {
                label: 'Historical Sales',
                data: [12000, 13500, 14200, 15000, 16300, 17200],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: false,
                pointRadius: 3
            },
            {
                label: 'Predicted Sales',
                data: [17200, 17800, 18600, 19200, 19700, 20300],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderDash: [5, 5],
                fill: false,
                pointRadius: 3
            },
            {
                label: 'Upper Bound',
                data: [17200, 18500, 19600, 20500, 21200, 22000],
                borderColor: 'rgba(255, 99, 132, 0.3)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                borderDash: [2, 2],
                pointRadius: 0,
                fill: '+1'
            },
            {
                label: 'Lower Bound',
                data: [17200, 17100, 17600, 17900, 18200, 18600],
                borderColor: 'rgba(255, 99, 132, 0.3)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                borderDash: [2, 2],
                pointRadius: 0,
                fill: false
            }
        ]
    };
    
    renderSalesPredictionChart(predictionData);
}

/**
 * Render sales prediction chart
 */
function renderSalesPredictionChart(predictionData) {
    const ctx = document.getElementById('salesPredictionChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: predictionData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return '$' + context.raw.toLocaleString();
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Sales Prediction (Next 6 Months)'
                }
            }
        }
    });
}

/**
 * Update charts based on selected time period
 */
function updateChartsTimePeriod(timePeriod) {
    // This would typically fetch new data for the selected time period
    console.log(`Updating charts for time period: ${timePeriod}`);
    
    // Show loading spinner
    document.querySelectorAll('.chart-container').forEach(container => {
        container.classList.add('loading');
    });
    
    // Simulate API call delay
    setTimeout(() => {
        // Remove loading spinner
        document.querySelectorAll('.chart-container').forEach(container => {
            container.classList.remove('loading');
        });
        
        // Reinitialize charts with new data
        initializeSalesTrendChart();
        initializeCategoryAnalysisChart();
        initializeSalesPredictionChart();
    }, 1000);
}

/**
 * Update category chart based on selected category
 */
function updateCategoryChart(category) {
    console.log(`Updating category chart for: ${category}`);
    
    // This would typically fetch new data for the selected category
    // For now, just simulate a refresh
    
    document.getElementById('categoryAnalysisChart').parentNode.classList.add('loading');
    
    setTimeout(() => {
        document.getElementById('categoryAnalysisChart').parentNode.classList.remove('loading');
        // If category is 'all', reinitialize with all categories
        // Otherwise filter for specific category
        initializeCategoryAnalysisChart();
    }, 1000);
}

/**
 * Refresh all dashboard data
 */
function refreshDashboardData() {
    console.log('Refreshing dashboard data');
    
    // Show loading spinner on all containers
    document.querySelectorAll('.data-container').forEach(container => {
        container.classList.add('loading');
    });
    
    // Reload all data
    loadSummaryStatistics();
    initializeSalesTrendChart();
    initializeCategoryAnalysisChart();
    initializeSalesPredictionChart();
    
    // Remove loading spinners after a delay
    setTimeout(() => {
        document.querySelectorAll('.data-container').forEach(container => {
            container.classList.remove('loading');
        });
    }, 1500);
}

/**
 * Toggle between different dashboard views
 */
function toggleView(viewType) {
    console.log(`Switching to view: ${viewType}`);
    
    // Hide all views
    document.querySelectorAll('.dashboard-view').forEach(view => {
        view.classList.add('d-none');
    });
    
    // Show selected view
    document.getElementById(`${viewType}View`).classList.remove('d-none');
    
    // Update active button
    document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`.view-toggle-btn[data-view="${viewType}"]`).classList.add('active');
}

/**
 * Load report images and handle their display
 */
function loadReportImages() {
    // Check for report images in the document
    document.querySelectorAll('.report-image').forEach(imgContainer => {
        const imgSrc = imgContainer.dataset.src;
        if (imgSrc) {
            const img = document.createElement('img');
            img.src = imgSrc;
            img.alt = imgContainer.dataset.title || 'Report Image';
            img.classList.add('img-fluid');
            
            // Add loading indicator
            imgContainer.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
            
            // Replace with image when loaded
            img.onload = function() {
                imgContainer.innerHTML = '';
                imgContainer.appendChild(img);
            };
            
            img.onerror = function() {
                imgContainer.innerHTML = '<div class="alert alert-warning">Image failed to load</div>';
            };
        }
    });
}
/**
 * Format a number as currency
 */
function formatCurrency(value) {
    if (typeof value !== 'number') {
        value = parseFloat(value) || 0;
    }
    return '$' + value.toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

/**
 * Show error message to user
 */
function showErrorMessage(message) {
    // Check if error container exists
    let errorContainer = document.getElementById('errorContainer');
    
    // Create one if it doesn't exist
    if (!errorContainer) {
        errorContainer = document.createElement('div');
        errorContainer.id = 'errorContainer';
        errorContainer.className = 'alert alert-danger alert-dismissible fade show';
        errorContainer.style.position = 'fixed';
        errorContainer.style.top = '20px';
        errorContainer.style.right = '20px';
        errorContainer.style.zIndex = '9999';
        
        document.body.appendChild(errorContainer);
    }
    
    // Set message
    errorContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        const alert = bootstrap.Alert.getOrCreateInstance(errorContainer);
        alert.close();
    }, 5000);
}

/**
 * Export dashboard data to CSV
 */
function exportDataToCSV() {
    // This would be implemented to export current chart/table data
    console.log('Exporting data to CSV');
    
    // For example implementation:
    fetch('/api/data/latest')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Convert data to CSV format
                const csvContent = convertToCSV(data.data);
                
                // Create a download link
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.setAttribute('href', url);
                link.setAttribute('download', 'sales_data.csv');
                link.style.visibility = 'hidden';
                
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            } else {
                showErrorMessage('Failed to export data');
            }
        })
        .catch(error => {
            console.error('Export error:', error);
            showErrorMessage('Error exporting data');
        });
}

/**
 * Convert JSON data to CSV format
 */
function convertToCSV(jsonData) {
    if (!jsonData || !jsonData.length) {
        return '';
    }
    
    const header = Object.keys(jsonData[0]).join(',') + '\n';
    const rows = jsonData.map(row => {
        return Object.values(row).map(value => {
            // Handle values with commas by quoting them
            if (typeof value === 'string' && value.includes(',')) {
                return `"${value}"`;
            }
            return value;
        }).join(',');
    }).join('\n');
    
    return header + rows;
}

/**
 * Handle printing of reports or dashboard views
 */
function printDashboard() {
    console.log('Printing dashboard view');
    window.print();
}

/**
 * Initialize date range picker
 */
function initializeDatePicker() {
    // If we're using a date range picker component
    const dateRangePicker = document.getElementById('dateRangePicker');
    
    if (dateRangePicker) {
        // Using a hypothetical date picker library
        // This could be replaced with actual code for a specific library
        $(dateRangePicker).daterangepicker({
            startDate: moment().subtract(30, 'days'),
            endDate: moment(),
            ranges: {
                'Last 7 Days': [moment().subtract(6, 'days'), moment()],
                'Last 30 Days': [moment().subtract(29, 'days'), moment()],
                'This Month': [moment().startOf('month'), moment().endOf('month')],
                'Last Month': [moment().subtract(1, 'month').startOf('month'), moment().subtract(1, 'month').endOf('month')],
                'Last 90 Days': [moment().subtract(89, 'days'), moment()],
                'Year to Date': [moment().startOf('year'), moment()]
            }
        }, function(start, end, label) {
            console.log(`Date range selected: ${start.format('YYYY-MM-DD')} to ${end.format('YYYY-MM-DD')}`);
            updateChartsDateRange(start.format('YYYY-MM-DD'), end.format('YYYY-MM-DD'));
        });
    }
}

/**
 * Update charts based on selected date range
 */
function updateChartsDateRange(startDate, endDate) {
    console.log(`Updating charts for date range: ${startDate} to ${endDate}`);
    
    // Show loading indicators
    document.querySelectorAll('.chart-container').forEach(container => {
        container.classList.add('loading');
    });
    
    // This would typically make API calls with the date range parameters
    // For now, simulate a delay and refresh
    setTimeout(() => {
        // Remove loading indicators
        document.querySelectorAll('.chart-container').forEach(container => {
            container.classList.remove('loading');
        });
        
        // Reinitialize charts with new data
        initializeSalesTrendChart();
        initializeCategoryAnalysisChart();
        initializeSalesPredictionChart();
    }, 1500);
}

/**
 * Handle modal dialog for prediction settings
 */
function showPredictionSettings() {
    // Get the modal element
    const modal = document.getElementById('predictionSettingsModal');
    
    if (modal) {
        // Using Bootstrap's modal
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }
}

/**
 * Apply prediction settings from modal
 */
function applyPredictionSettings() {
    // Get form values
    const predictionDays = document.getElementById('predictionDays').value;
    const confidenceInterval = document.getElementById('confidenceInterval').value;
    const modelType = document.querySelector('input[name="modelType"]:checked').value;
    
    console.log(`Applying prediction settings: ${predictionDays} days, ${confidenceInterval} confidence, model: ${modelType}`);
    
    // Close modal
    const modal = document.getElementById('predictionSettingsModal');
    if (modal) {
        const bsModal = bootstrap.Modal.getInstance(modal);
        bsModal.hide();
    }
    
    // Update prediction chart with new settings
    document.getElementById('salesPredictionChart').parentNode.classList.add('loading');
    
    // This would typically make an API call with the new settings
    // For now, simulate a delay and refresh
    setTimeout(() => {
        document.getElementById('salesPredictionChart').parentNode.classList.remove('loading');
        initializeSalesPredictionChart();
    }, 1500);
}

/**
 * Initialize interactive data table
 */
function initializeDataTable() {
    const dataTable = document.getElementById('salesDataTable');
    
    if (dataTable) {
        // Using a hypothetical data table library
        // This could be replaced with actual code for a specific library
        $(dataTable).DataTable({
            processing: true,
            serverSide: false, // For demo, we'll load all data at once
            searching: true,
            ordering: true,
            paging: true,
            lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
            ajax: {
                url: '/api/data/latest',
                dataSrc: 'data'
            },
            columns: [
                { data: 'date' },
                { 
                    data: 'sales',
                    render: function(data) {
                        return formatCurrency(data);
                    }
                },
                { data: 'category' },
                { data: 'region' }
            ]
        });
    }
}

/**
 * Save dashboard configuration
 */
function saveDashboardConfig() {
    // Get current dashboard configuration
    const config = {
        charts: {
            salesTrend: document.getElementById('salesTrendChart') ? true : false,
            categoryAnalysis: document.getElementById('categoryAnalysisChart') ? true : false,
            salesPrediction: document.getElementById('salesPredictionChart') ? true : false
        },
        timeFrame: document.getElementById('timePeriodSelector') ? 
                   document.getElementById('timePeriodSelector').value : '30days',
        layout: document.querySelector('.dashboard-layout') ? 
                document.querySelector('.dashboard-layout').dataset.layout : 'grid'
    };
    
    console.log('Saving dashboard configuration:', config);
    
    // Save to localStorage
    localStorage.setItem('dashboardConfig', JSON.stringify(config));
    
    // Show confirmation
    showSuccessMessage('Dashboard configuration saved');
}

/**
 * Load dashboard configuration
 */
function loadDashboardConfig() {
    // Check if saved configuration exists
    const savedConfig = localStorage.getItem('dashboardConfig');
    
    if (savedConfig) {
        const config = JSON.parse(savedConfig);
        console.log('Loading dashboard configuration:', config);
        
        // Apply configuration
        if (document.getElementById('timePeriodSelector')) {
            document.getElementById('timePeriodSelector').value = config.timeFrame;
        }
        
        if (document.querySelector('.dashboard-layout')) {
            document.querySelector('.dashboard-layout').dataset.layout = config.layout;
            document.querySelector('.dashboard-layout').className = 
                'dashboard-layout ' + config.layout + '-layout';
        }
        
        // Update UI based on saved preferences
        updateChartsTimePeriod(config.timeFrame);
    }
}

/**
 * Show success message to user
 */
function showSuccessMessage(message) {
    // Check if success container exists
    let successContainer = document.getElementById('successContainer');
    
    // Create one if it doesn't exist
    if (!successContainer) {
        successContainer = document.createElement('div');
        successContainer.id = 'successContainer';
        successContainer.className = 'alert alert-success alert-dismissible fade show';
        successContainer.style.position = 'fixed';
        successContainer.style.top = '20px';
        successContainer.style.right = '20px';
        successContainer.style.zIndex = '9999';
        
        document.body.appendChild(successContainer);
    }
    
    // Set message
    successContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Auto hide after 3 seconds
    setTimeout(() => {
        const alert = bootstrap.Alert.getOrCreateInstance(successContainer);
        alert.close();
    }, 3000);
}

