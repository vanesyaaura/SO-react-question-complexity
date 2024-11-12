let crawlingInProgress = false;

function crawlData() {
    if (crawlingInProgress) {
        alert('Crawling is already in progress. Please wait.');
        return;
    }

    const statusDiv = document.getElementById('crawl-status');
    statusDiv.innerHTML = '<div class="alert alert-info">Crawling data... Please wait.</div>';
    crawlingInProgress = true;

    fetch('/crawl', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        statusDiv.innerHTML = '<div class="alert alert-success">Data crawling started. This may take several minutes...</div>';
    })
    .catch(error => {
        statusDiv.innerHTML = '<div class="alert alert-danger">Error during data crawling. Please try again.</div>';
        console.error('Error:', error);
    })
    .finally(() => {
        crawlingInProgress = false;
    });
}

let analysisInProgress = false;

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