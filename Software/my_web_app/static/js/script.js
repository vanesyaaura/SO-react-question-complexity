let crawlingInProgress = false;
let isPaused = false;
let statusCheckInterval = null;

function updateButtonStates(isRunning) {
    document.getElementById('crawl-btn').disabled = isRunning;
    document.getElementById('control-buttons').style.display = isRunning ? 'flex' : 'none';
}

function updateProgress(progress, currentFile) {
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const statusDiv = document.getElementById('crawl-status');
    
    progressContainer.style.display = 'block';
    progressBar.style.width = `${progress}%`;
    progressBar.textContent = `${Math.round(progress)}%`;
    
    if (currentFile) {
        statusDiv.innerHTML = `<div class="alert alert-info">Processing: ${currentFile}</div>`;
    }
}

function checkCrawlStatus() {
    fetch('/crawl_status')
        .then(response => response.json())
        .then(status => {
            crawlingInProgress = status.is_running;
            isPaused = status.is_paused;
            updateButtonStates(status.is_running);
            updatePauseResumeButton();
            
            if (status.is_running) {
                updateProgress(status.progress, status.current_file);
            } else {
                clearInterval(statusCheckInterval);
                document.getElementById('progress-container').style.display = 'none';
                if (!status.is_paused) {
                    document.getElementById('crawl-status').innerHTML = 
                        '<div class="alert alert-success">Crawling completed successfully!</div>';
                    document.getElementById('control-buttons').style.display = 'none';
                }
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            clearInterval(statusCheckInterval);
        });
}

function updatePauseResumeButton() {
    const button = document.getElementById('pause-resume-btn');
    const icon = document.getElementById('pause-resume-icon');
    const text = document.getElementById('pause-resume-text');
    
    if (isPaused) {
        button.classList.remove('btn-warning');
        button.classList.add('btn-success');
        icon.src = 'static/images/resume.png';
        icon.alt = 'Resume';
        text.textContent = 'Resume';
    } else {
        button.classList.remove('btn-success');
        button.classList.add('btn-warning');
        icon.src = 'static/images/pause.png';
        icon.alt = 'Pause';
        text.textContent = 'Pause';
    }
}

function togglePauseResume() {
    const action = isPaused ? 'resume_crawl' : 'pause_crawl';
    
    fetch(`/${action}`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            isPaused = !isPaused;
            updatePauseResumeButton();
            
            document.getElementById('crawl-status').innerHTML = 
                `<div class="alert alert-info">Crawling ${isPaused ? 'paused' : 'resumed'}</div>`;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('crawl-status').innerHTML = 
                `<div class="alert alert-danger">Error: ${error.message}</div>`;
        });
}

function crawlData() {
    if (crawlingInProgress) {
        alert('Crawling is already in progress. Please wait.');
        return;
    }

    const statusDiv = document.getElementById('crawl-status');
    statusDiv.innerHTML = '<div class="alert alert-info">Starting crawl process...</div>';
    
    fetch('/crawl', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        statusDiv.innerHTML = '<div class="alert alert-info">Data crawling started...</div>';
        crawlingInProgress = true;
        isPaused = false;
        updateButtonStates(true);
        updatePauseResumeButton();
        
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        statusCheckInterval = setInterval(checkCrawlStatus, 1000);
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        console.error('Error:', error);
    });
}

function stopCrawl() {
    if (!confirm('Are you sure you want to stop the crawling process? This cannot be undone.')) {
        return;
    }

    fetch('/stop_crawl', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            document.getElementById('crawl-status').innerHTML = 
                '<div class="alert alert-info">Crawling stopped</div>';
            crawlingInProgress = false;
            isPaused = false;
            updateButtonStates(false);
            clearInterval(statusCheckInterval);
            document.getElementById('progress-container').style.display = 'none';
            document.getElementById('control-buttons').style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('crawl-status').innerHTML = 
                `<div class="alert alert-danger">Error: ${error.message}</div>`;
        });
}

let analysisInProgress = false;

function uploadDatasets() {
    const file1 = document.getElementById('datasetInput1').files[0];
    const file2 = document.getElementById('datasetInput2').files[0];
    const uploadStatus = document.getElementById('upload-status');

    if (!file1 || !file2) {
        uploadStatus.innerHTML = '<div class="alert alert-danger">Please select both datasets</div>';
        return;
    }

    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    uploadStatus.innerHTML = '<div class="alert alert-info">Uploading datasets...</div>';

    fetch('/upload_dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        uploadStatus.innerHTML = '<div class="alert alert-success">Datasets uploaded successfully!</div>';
    })
    .catch(error => {
        uploadStatus.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        console.error('Error:', error);
    });
}

function startAnalysis() {
    if (analysisInProgress) {
        alert('Analysis is already in progress. Please wait.');
        return;
    }

    const analysisType = document.getElementById("analysisType").value;
    const resultsContainer = document.getElementById("results");
    const plotContainer = document.getElementById("plot-container");
    const wordFreqContainer = document.getElementById("word-freq-container");
    const coefTableContainer = document.getElementById("coef-table-container");
    const coefTable = document.getElementById("coef-table");

    resultsContainer.innerHTML = '<div class="alert alert-info">Processing analysis...</div>';
    plotContainer.style.display = "none";
    wordFreqContainer.style.display = "none";
    coefTableContainer.style.display = "none";
    analysisInProgress = true;

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "analysis_type": analysisType })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        if (analysisType === "logistic_regression") {
            return response.text();
        }
        return response.json();
    })
    .then(data => {
        resultsContainer.innerHTML = '';

        if (analysisType === "linear_regression") {
            const plotImage = document.getElementById("analysis-plot");
            plotImage.src = data.plot_url;
            plotContainer.style.display = "block";

            resultsContainer.innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <h3>Linear Regression Results</h3>
                        <p><strong>Coefficient:</strong> ${Number(data.coef).toFixed(4)}</p>
                        <p><strong>Intercept:</strong> ${Number(data.intercept).toFixed(4)}</p>
                    </div>
                </div>`;
        }
        else if (analysisType === "logistic_regression") {
            coefTable.innerHTML = data;
            coefTableContainer.style.display = "block";
        }
    })
    .catch(error => {
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                Error during analysis: ${error.message}
                Please try again.
            </div>`;
        console.error('Error:', error);
    })
    .finally(() => {
        analysisInProgress = false;
    });
}

document.addEventListener('DOMContentLoaded', function() {
    updateButtonStates(false, false);
    
    document.getElementById('progress-container').style.display = 'none';
});