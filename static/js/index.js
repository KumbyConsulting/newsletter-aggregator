// Topic Distribution Charts
let topicChart = null;
let topicBarChart = null;

// Function to toggle article content visibility
function toggleArticleContent(button) {
    const contentContainer = button.closest('div').parentElement;
    const fullContent = contentContainer.querySelector('.article-full-content');
    
    if (fullContent.classList.contains('hidden')) {
        fullContent.classList.remove('hidden');
        button.textContent = 'Hide Full Content';
    } else {
        fullContent.classList.add('hidden');
        button.textContent = 'Show Full Content';
    }
}

// Helper function to strip HTML tags for safe display
function stripHtmlTags(html) {
    if (!html) return '';
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;
    return tempDiv.textContent || tempDiv.innerText || '';
}

// Function to create pie chart
function createPieChart() {
    const ctx = document.getElementById('topicChart')?.getContext('2d');
    if (!ctx) return;

    const topics = Array.from(document.querySelectorAll('#tableView tbody tr')).map(row => ({
        topic: row.cells[0].textContent.trim(),
        count: parseInt(row.cells[1].textContent),
        percentage: parseFloat(row.cells[2].textContent)
    }));

    if (topicChart) {
        topicChart.destroy();
    }

    const colors = [
        '#00405e', '#7f9360', '#f9e15e', '#f1f2e7',
        '#2a5a7a', '#8fa67a', '#fae67e', '#f4f5e9',
        '#4a7a9a', '#afb69a', '#fae89e', '#f7f8e9'
    ];

    // Update legend colors
    document.querySelectorAll('.chart-color').forEach((el, index) => {
        el.style.backgroundColor = colors[index % colors.length];
    });

    topicChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: topics.map(t => t.topic),
            datasets: [{
                data: topics.map(t => t.count),
                backgroundColor: colors,
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const topic = topics[context.dataIndex];
                            return `${topic.topic}: ${topic.count} (${topic.percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Function to create bar chart
function createBarChart() {
    const ctx = document.getElementById('topicBarChart')?.getContext('2d');
    if (!ctx) return;

    const topics = Array.from(document.querySelectorAll('#tableView tbody tr')).map(row => ({
        topic: row.cells[0].textContent.trim(),
        count: parseInt(row.cells[1].textContent),
        percentage: parseFloat(row.cells[2].textContent)
    }));

    if (topicBarChart) {
        topicBarChart.destroy();
    }

    topicBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: topics.map(t => t.topic),
            datasets: [{
                label: 'Number of Articles',
                data: topics.map(t => t.count),
                backgroundColor: '#00405e',
                borderColor: '#00405e',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Articles'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Topics'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const topic = topics[context.dataIndex];
                            return `Articles: ${topic.count} (${topic.percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Function to initialize progress bars with animation
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const percentage = bar.getAttribute('data-percentage');
        if (percentage) {
            // Set initial width to 0
            bar.style.width = '0%';
            bar.style.backgroundColor = '#00405e';
            
            // Trigger animation after a small delay
            setTimeout(() => {
                bar.style.width = percentage + '%';
            }, 100);
        }
    });
}

// Function to toggle between views
function toggleView(viewType) {
    const views = ['table', 'chart', 'bar'];
    const buttons = document.querySelectorAll('.view-toggle-btn');
    const viewElements = document.querySelectorAll('.view-content');

    // Update button states
    buttons.forEach(btn => {
        const isActive = btn.getAttribute('data-view') === viewType;
        btn.setAttribute('aria-pressed', isActive);
        btn.classList.toggle('bg-primary', isActive);
        btn.classList.toggle('text-white', isActive);
    });

    // Show/hide views
    viewElements.forEach(el => {
        const isTargetView = el.id === `${viewType}View`;
        el.classList.toggle('hidden', !isTargetView);
    });

    // Initialize appropriate chart if needed
    if (viewType === 'chart' && !topicChart) {
        createPieChart();
    } else if (viewType === 'bar' && !topicBarChart) {
        createBarChart();
    }
}

// Initialize topic distribution functionality
function initializeTopicDistribution() {
    // Initialize progress bars
    initializeProgressBars();

    // Add click handlers for view toggle buttons
    document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            toggleView(btn.getAttribute('data-view'));
        });
    });

    // Add click handlers for chart legend
    document.querySelectorAll('#chartLegend .chart-color').forEach((el, index) => {
        el.addEventListener('click', () => {
            if (topicChart) {
                const meta = topicChart.getDatasetMeta(0);
                const activeState = !meta.data[index].hidden;
                meta.data[index].hidden = activeState;
                el.parentElement.classList.toggle('opacity-50', activeState);
                topicChart.update();
            }
        });
    });
}

async function showSimilarArticles(articleId) {
    try {
        const response = await fetch(`/similar-articles/${articleId}`);
        const data = await response.json();
        
        // Create modal content
        let modalContent = `
        <div class="bg-white p-6 rounded-lg max-w-2xl mx-auto">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-2xl font-bold">Similar Articles</h2>
                <button class="text-gray-500 hover:text-gray-700" onclick="this.closest('.fixed').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="space-y-4">`;
            
        if (data.articles.length === 0) {
            modalContent += `<p class="text-gray-600">No similar articles found.</p>`;
        } else {
            data.articles.forEach(article => {
                modalContent += `
                    <div class="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
                        <h3 class="font-bold text-lg mb-1">${article.metadata.title}</h3>
                        <div class="flex justify-between text-sm text-gray-600 mb-2">
                            <span>${article.metadata.source}</span>
                            <span>${article.metadata.pub_date}</span>
                        </div>
                        <div class="mb-3">
                            <span class="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                                ${article.metadata.topic || 'Uncategorized'}
                            </span>
                        </div>
                        <a href="${article.metadata.link}" 
                           class="text-blue-600 hover:text-blue-800 inline-flex items-center"
                           target="_blank">
                            Read Article <i class="fas fa-external-link-alt ml-1"></i>
                        </a>
                    </div>`;
            });
        }
        
        modalContent += `</div></div>`;
        
        // Show modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50';
        modal.innerHTML = modalContent;
        modal.onclick = e => {
            if (e.target === modal) modal.remove();
        };
        document.body.appendChild(modal);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error fetching similar articles');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function shareArticle(title, url) {
    if (navigator.share) {
        navigator.share({
            title: title,
            url: url
        }).catch(console.error);
    } else {
        // Fallback copy to clipboard
        navigator.clipboard.writeText(url).then(() => {
            // Show toast notification
            const toast = document.createElement('div');
            toast.className = 'fixed bottom-4 right-4 bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg z-50';
            toast.innerHTML = '<i class="fas fa-check-circle mr-2"></i> Link copied to clipboard!';
            document.body.appendChild(toast);
            
            // Remove toast after 3 seconds
            setTimeout(() => {
                toast.remove();
            }, 3000);
        }).catch(console.error);
    }
}

// Initialize infinite scroll
let isLoading = false;
let hasMore = true;
let scrollDebounceTimer = null;
const loadingIndicator = document.getElementById('loadingIndicator');

// Update scroll check function for better performance
function checkScrollPosition() {
    if (scrollDebounceTimer) {
        clearTimeout(scrollDebounceTimer);
    }
    
    scrollDebounceTimer = setTimeout(() => {
        if (isLoading || !hasMore) return;
        
        const scrollPosition = window.innerHeight + window.scrollY;
        const bodyHeight = document.body.offsetHeight;
        const scrollThreshold = bodyHeight * 0.8; // Load when 80% scrolled
        
        if (scrollPosition >= scrollThreshold) {
            loadMoreArticles();
        }
    }, 150); // Slightly increased debounce time for better performance
}

// Helper function to safely get article data with fallbacks
function getArticleData(article, field, defaultValue = '') {
    try {
        if (!article || !article.metadata) {
            console.warn('Article or metadata is missing:', article);
            return defaultValue;
        }
        return article.metadata[field] || defaultValue;
    } catch (error) {
        console.error(`Error getting ${field} from article:`, error);
        return defaultValue;
    }
}

function createArticleCard(article) {
    try {
        if (!article) {
            console.error('Received null or undefined article');
            return null;
        }

        console.log('Creating card for article:', article);  // Debug log

        const card = document.createElement('div');
        card.className = 'card-custom article-card';
        
        // Get article data with fallbacks
        const title = getArticleData(article, 'title', 'Untitled');
        const description = getArticleData(article, 'description', 'No description available');
        const link = getArticleData(article, 'link', '#');
        const pubDate = getArticleData(article, 'pub_date', 'Date unknown');
        const topic = getArticleData(article, 'topic', 'Uncategorized');
        const source = getArticleData(article, 'source', 'Unknown source');
        const summary = getArticleData(article, 'summary');
        const imageUrl = getArticleData(article, 'image_url', '');
        const hasFullContent = Boolean(getArticleData(article, 'has_full_content'));
        const readingTime = getArticleData(article, 'reading_time');
        
        // Create image or placeholder
        const hasImage = imageUrl && imageUrl.trim() !== '';
        
        // Build HTML content with proper handling of full content
        let descriptionHtml = '';
        const truncatedDescription = stripHtmlTags(description).substring(0, 200);
        
        if (hasFullContent) {
            descriptionHtml = `
                <div class="mb-4">
                    <div class="flex justify-between items-center mb-2">
                        <h4 class="font-bold text-sm uppercase tracking-wider text-gray-600">Full Article</h4>
                        <button class="text-blue-600 hover:text-blue-800 text-sm" 
                                onclick="toggleArticleContent(this)">
                            Show Full Content
                        </button>
                    </div>
                    <p class="text-gray-700 article-card__description">${truncatedDescription}...</p>
                    <div class="article-full-content hidden mt-2 max-h-96 overflow-y-auto">
                    </div>
                </div>
            `;
        } else {
            descriptionHtml = `
                <div class="mb-4">
                    <p class="text-gray-700 article-card__description">${truncatedDescription}...</p>
                </div>
            `;
        }
        
        card.innerHTML = `
            <div class="article-card__image ${!hasImage ? 'article-card__image-placeholder' : ''}" 
                 ${hasImage ? `style="background-image: url('${imageUrl}');"` : ''}>
                ${!hasImage ? '<i class="fas fa-newspaper text-white text-4xl"></i>' : ''}
                <div class="article-card__topic">
                    <span class="topic-badge">
                        ${topic}
                    </span>
                </div>
            </div>
            <div class="p-6 article-card__content">
                <div class="flex flex-wrap items-center text-sm text-gray-600 mb-3">
                    <div class="mr-4 mb-2">
                        <i class="fas fa-calendar-alt mr-1"></i>
                        <span>${pubDate}</span>
                    </div>
                    <div class="mr-4 mb-2">
                        <i class="fas fa-newspaper mr-1"></i> ${source}
                    </div>
                    ${readingTime ? `
                    <div class="mr-4 mb-2">
                        <i class="far fa-clock mr-1"></i> ${readingTime} min read
                    </div>
                    ` : ''}
                </div>
                <h3 class="text-xl font-bold mb-3 article-card__title">${title}</h3>
                ${descriptionHtml}
                ${summary ? `
                <div class="bg-gray-50 p-4 rounded-lg mb-4 border-l-4 border-blue-500">
                    <h4 class="font-bold mb-2 text-sm uppercase tracking-wider text-gray-600">Summary</h4>
                    <p class="text-sm text-gray-700">${summary}</p>
                </div>
                ` : ''}
                <div class="article-card__footer flex justify-between items-center">
                    <a href="${link}" 
                       target="_blank" 
                       class="btn-custom px-4 py-2 rounded-lg text-sm">
                        Read Article <i class="fas fa-external-link-alt ml-1"></i>
                    </a>
                    <div class="flex space-x-1">
                        ${article.id ? `
                        <button onclick="showSimilarArticles('${article.id}')"
                                class="action-btn text-gray-600 hover:text-gray-800" 
                                title="Find similar articles">
                            <i class="fas fa-layer-group"></i>
                        </button>
                        ` : ''}
                        <button onclick="shareArticle('${title}', '${link}')"
                                class="action-btn text-gray-600 hover:text-gray-800"
                                title="Share article">
                            <i class="fas fa-share-alt"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Set full content HTML safely after the card is created
        if (hasFullContent) {
            const fullContentDiv = card.querySelector('.article-full-content');
            if (fullContentDiv) {
                fullContentDiv.innerHTML = description;
            }
        }
        
        return card;
    } catch (error) {
        console.error('Error creating article card:', error);
        return null;
    }
}

// Debounce function for search suggestions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Search suggestions functionality
async function fetchSearchSuggestions(query) {
    if (!query || query.length < 2) {
        document.getElementById('searchSuggestions').classList.add('hidden');
        return;
    }

    try {
        const response = await fetch(`/api/search/suggestions?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        const suggestionsDiv = document.getElementById('searchSuggestions');
        if (data.suggestions && data.suggestions.length > 0) {
            suggestionsDiv.innerHTML = data.suggestions.map(suggestion => `
                <button class="w-full text-left px-4 py-2 hover:bg-gray-100 focus:bg-gray-100 focus:outline-none" 
                        onclick="selectSearchSuggestion('${suggestion}')">
                    ${suggestion}
                </button>
            `).join('');
            suggestionsDiv.classList.remove('hidden');
        } else {
            suggestionsDiv.classList.add('hidden');
        }
    } catch (error) {
        console.error('Error fetching suggestions:', error);
    }
}

// Select search suggestion
function selectSearchSuggestion(suggestion) {
    document.getElementById('search').value = suggestion;
    document.getElementById('searchSuggestions').classList.add('hidden');
    document.getElementById('searchForm').submit();
}

// Clear search form
function clearSearchForm() {
    const form = document.getElementById('searchForm');
    const searchInput = document.getElementById('search');
    const topicSelect = document.getElementById('topic');
    
    searchInput.value = '';
    topicSelect.value = 'All';
    form.submit();
}

// Toggle sort order
function toggleSortOrder() {
    const btn = document.getElementById('sortOrderBtn');
    const input = document.getElementById('sort_order');
    const currentOrder = btn.getAttribute('data-order');
    const newOrder = currentOrder === 'asc' ? 'desc' : 'asc';
    
    btn.setAttribute('data-order', newOrder);
    input.value = newOrder;
    
    // Update icon
    const icon = btn.querySelector('i');
    icon.className = `fas fa-sort-amount-${newOrder === 'asc' ? 'up' : 'down'}`;
    
    // Submit form
    document.getElementById('searchForm').submit();
}

// Initialize all event listeners and functionality
function initializeApp() {
    // Mobile menu functionality
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileMenu = document.getElementById('mobileMenu');
    
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
            const isExpanded = mobileMenuBtn.getAttribute('aria-expanded') === 'true';
            mobileMenuBtn.setAttribute('aria-expanded', !isExpanded);
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', (event) => {
            if (!mobileMenuBtn.contains(event.target) && !mobileMenu.contains(event.target)) {
                mobileMenu.classList.add('hidden');
                mobileMenuBtn.setAttribute('aria-expanded', 'false');
            }
        });

        // Mobile menu button handlers
        document.getElementById('mobileUpdateNewsBtn')?.addEventListener('click', () => {
            mobileMenu.classList.add('hidden');
            document.getElementById('updateNewsBtn')?.click();
        });

        document.getElementById('mobileExportBtn')?.addEventListener('click', () => {
            mobileMenu.classList.add('hidden');
            document.getElementById('exportToCsvBtn')?.click();
        });

        document.getElementById('mobileForceSyncBtn')?.addEventListener('click', () => {
            mobileMenu.classList.add('hidden');
            document.getElementById('forceSyncBtn')?.click();
        });

        document.getElementById('mobileCleanupBtn')?.addEventListener('click', () => {
            mobileMenu.classList.add('hidden');
            document.getElementById('cleanupDuplicatesBtn')?.click();
        });
    }

    // Initialize chart if needed
    if (document.getElementById('chartView') && !document.getElementById('chartView').classList.contains('hidden')) {
        createPieChart();
    }

    // Initialize progress bars for topic distribution
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const percentage = bar.getAttribute('data-percentage');
        if (percentage) {
            bar.style.width = percentage + '%';
        }
    });

    // Add event delegation for action buttons
    document.addEventListener('click', function(event) {
        const button = event.target.closest('button[data-action]');
        if (!button) return;

        const action = button.getAttribute('data-action');
        
        switch (action) {
            case 'similar-articles':
                const articleId = button.getAttribute('data-article-id');
                showSimilarArticles(articleId);
                break;
            
            case 'share-article':
                const title = escapeHtml(button.getAttribute('data-article-title'));
                const url = escapeHtml(button.getAttribute('data-article-url'));
                shareArticle(title, url);
                break;
        }
    });

    // Initialize update functionality
    initializeUpdateFunctionality();
    
    // Initialize infinite scroll
    window.addEventListener('scroll', checkScrollPosition);
    
    // Initialize search form loading indicator
    const searchForm = document.getElementById('searchForm');
    const searchLoading = document.getElementById('searchLoading');
    
    if (searchForm && searchLoading) {
        searchForm.addEventListener('submit', function() {
            searchLoading.classList.remove('hidden');
        });
    }

    // Initialize search functionality
    const searchInput = document.getElementById('search');
    if (searchInput) {
        const debouncedFetch = debounce(fetchSearchSuggestions, 300);
        searchInput.addEventListener('input', (e) => debouncedFetch(e.target.value));
        
        // Close suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target)) {
                document.getElementById('searchSuggestions').classList.add('hidden');
            }
        });
    }

    // Initialize clear search button
    const clearSearchBtn = document.getElementById('clearSearch');
    if (clearSearchBtn) {
        clearSearchBtn.addEventListener('click', clearSearchForm);
    }

    // Initialize sort order toggle
    const sortOrderBtn = document.getElementById('sortOrderBtn');
    if (sortOrderBtn) {
        sortOrderBtn.addEventListener('click', toggleSortOrder);
    }

    // Initialize topic select auto-submit
    const topicSelect = document.getElementById('topic');
    if (topicSelect) {
        topicSelect.addEventListener('change', () => {
            document.getElementById('searchForm').submit();
        });
    }

    // Initialize sort by select auto-submit
    const sortBySelect = document.getElementById('sort_by');
    if (sortBySelect) {
        sortBySelect.addEventListener('change', () => {
            document.getElementById('searchForm').submit();
        });
    }
}

// Call initialization when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);

// Initialize update functionality
function initializeUpdateFunctionality() {
    const updateBtn = document.getElementById('updateNewsBtn');
    const updateLoader = document.getElementById('updateNewsLoader');
    const updateStatusBar = document.getElementById('updateStatusBar');
    const closeUpdateStatus = document.getElementById('closeUpdateStatus');
    const continueButton = document.getElementById('continueButton');
    
    // Progress elements in loader
    const loaderProgressBar = document.getElementById('updateLoaderProgressBar');
    const loaderMessage = document.getElementById('updateLoaderMessage');
    const sourcesProcessed = document.getElementById('sourcesProcessed');
    const totalSources = document.getElementById('totalSources');
    const articlesFound = document.getElementById('articlesFound');
    
    // Progress elements in status bar
    const statusProgressBar = document.getElementById('updateProgressBar');
    const statusMessage = document.getElementById('updateStatusMessage');
    const statusTitle = document.getElementById('updateStatusTitle');
    const sourcesProcessedText = document.getElementById('sourcesProcessedText');
    const articlesFoundText = document.getElementById('articlesFoundText');
    
    // Function to update both progress indicators
    function updateProgressIndicators(data) {
        // Update loader
        if (loaderProgressBar) loaderProgressBar.style.width = data.progress + '%';
        if (loaderMessage) loaderMessage.textContent = data.message;
        if (sourcesProcessed) sourcesProcessed.textContent = data.sources_processed;
        if (totalSources) totalSources.textContent = data.total_sources;
        if (articlesFound) articlesFound.textContent = data.articles_found;
        
        // Update status bar
        if (statusProgressBar) statusProgressBar.style.width = data.progress + '%';
        if (statusMessage) statusMessage.textContent = data.message;
        if (sourcesProcessedText) sourcesProcessedText.textContent = 
            `${data.sources_processed} of ${data.total_sources} sources processed`;
        if (articlesFoundText) articlesFoundText.textContent = 
            `${data.articles_found} articles found`;
        
        // Update title and icon based on status
        const statusIcon = document.getElementById('updateStatusIcon');
        
        if (statusTitle) {
            if (data.in_progress) {
                if (statusIcon) statusIcon.classList.add('animate-spin');
            } else {
                if (statusIcon) statusIcon.classList.remove('animate-spin');
                
                if (data.status === 'completed') {
                    statusTitle.textContent = 'Update Complete';
                    statusTitle.classList.add('text-green-600');
                    if (statusIcon) {
                        statusIcon.className = 'fas fa-check-circle mr-2 text-green-500';
                    }
                } else if (data.status === 'failed') {
                    statusTitle.textContent = 'Update Failed';
                    statusTitle.classList.add('text-red-600');
                    if (statusIcon) {
                        statusIcon.className = 'fas fa-times-circle mr-2 text-red-500';
                    }
                } else if (data.status === 'completed_with_warnings' || data.status === 'completed_with_errors') {
                    statusTitle.textContent = 'Update Completed with Warnings';
                    statusTitle.classList.add('text-yellow-600');
                    if (statusIcon) {
                        statusIcon.className = 'fas fa-exclamation-triangle mr-2 text-yellow-500';
                    }
                }
            }
        }
    }
    
    // Function to poll update status
    let statusInterval = null;
    function pollUpdateStatus() {
        fetch('/api/update/status')
            .then(response => response.json())
            .then(data => {
                updateProgressIndicators(data);
                
                // If update is complete, stop polling
                if (!data.in_progress && data.status !== 'idle') {
                    if (statusInterval) {
                        clearInterval(statusInterval);
                        statusInterval = null;
                    }
                    
                    // Show completion message
                    if (data.status === 'completed') {
                        const toast = document.createElement('div');
                        toast.className = 'fixed bottom-4 right-4 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg z-50';
                        toast.innerHTML = '<i class="fas fa-check-circle mr-2"></i> Update completed successfully!';
                        document.body.appendChild(toast);
                        
                        setTimeout(() => {
                            toast.remove();
                        }, 5000);
                    }
                }
            })
            .catch(error => {
                console.error('Error polling update status:', error);
            });
    }
    
    // Start update process
    function startUpdate() {
        fetch('/api/update/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Show status bar
                updateStatusBar.classList.remove('hidden');
                
                // Start polling for updates
                if (statusInterval) {
                    clearInterval(statusInterval);
                }
                statusInterval = setInterval(pollUpdateStatus, 2000);
                
                // Initial poll
                pollUpdateStatus();
            } else {
                // Show error message
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error starting update:', error);
            alert('Error starting update. Please try again.');
        });
    }
    
    // Event listeners
    if (updateBtn) {
        updateBtn.addEventListener('click', function() {
            // Show loader
            if (updateLoader) {
                updateLoader.classList.remove('hidden');
            }
            
            // Initialize values
            if (totalSources) totalSources.textContent = '...';
            if (sourcesProcessed) sourcesProcessed.textContent = '0';
            if (articlesFound) articlesFound.textContent = '0';
            if (loaderProgressBar) loaderProgressBar.style.width = '0%';
            
            // Start update process
            startUpdate();
        });
    }
    
    // Continue button - hide loader but keep status bar
    if (continueButton) {
        continueButton.addEventListener('click', function() {
            if (updateLoader) {
                updateLoader.classList.add('hidden');
            }
        });
    }
    
    // Close status bar
    if (closeUpdateStatus) {
        closeUpdateStatus.addEventListener('click', function() {
            if (updateStatusBar) {
                updateStatusBar.classList.add('hidden');
            }
            
            // Stop polling if active
            if (statusInterval) {
                clearInterval(statusInterval);
                statusInterval = null;
            }
        });
    }
    
    // Check if update is in progress on page load
    fetch('/api/update/status')
        .then(response => response.json())
        .then(data => {
            if (data.in_progress) {
                // Show status bar
                if (updateStatusBar) {
                    updateStatusBar.classList.remove('hidden');
                }
                
                // Start polling
                statusInterval = setInterval(pollUpdateStatus, 2000);
                
                // Initial update
                updateProgressIndicators(data);
            }
        })
        .catch(error => {
            console.error('Error checking update status:', error);
        });
}

// Storage admin functionality
function initializeStorageAdmin() {
    const adminDropdownBtn = document.getElementById('adminDropdownBtn');
    const adminDropdownMenu = document.getElementById('adminDropdownMenu');

    if (adminDropdownBtn && adminDropdownMenu) {
        adminDropdownBtn.addEventListener('click', function() {
            adminDropdownMenu.classList.toggle('hidden');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(event) {
            if (!adminDropdownBtn.contains(event.target) && !adminDropdownMenu.contains(event.target)) {
                adminDropdownMenu.classList.add('hidden');
            }
        });
        
        // Sync storage button
        const exportToCsvBtn = document.getElementById('exportToCsvBtn');
        if (exportToCsvBtn) {
            exportToCsvBtn.addEventListener('click', function() {
                adminDropdownMenu.classList.add('hidden');
                exportToCsv(false);
            });
        }
        
        // Force sync button
        const forceSyncBtn = document.getElementById('forceSyncBtn');
        if (forceSyncBtn) {
            forceSyncBtn.addEventListener('click', function() {
                adminDropdownMenu.classList.add('hidden');
                exportToCsv(true);
            });
        }
        
        // Clean up duplicates button
        const cleanupDuplicatesBtn = document.getElementById('cleanupDuplicatesBtn');
        if (cleanupDuplicatesBtn) {
            cleanupDuplicatesBtn.addEventListener('click', function() {
                cleanupDuplicates();
            });
        }
    }
}

// Function to export articles to CSV
function exportToCsv(force = false) {
    // Show status
    const statusBar = document.getElementById('updateStatusBar');
    const statusTitle = document.getElementById('updateStatusTitle');
    const statusMessage = document.getElementById('updateStatusMessage');
    
    if (statusBar && statusTitle && statusMessage) {
        statusBar.classList.remove('hidden');
        statusTitle.textContent = 'Exporting Articles';
        statusMessage.textContent = 'Exporting articles from database to CSV...';
        
        // Make API call
        fetch('/api/storage/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ limit: 0 })
        })
        .then(response => response.json())
        .then(data => {
            // Update status
            statusTitle.textContent = data.success ? 'Export Complete' : 'Export Warning';
            statusMessage.textContent = data.message;
            
            // Change icon based on result
            const icon = statusTitle.previousElementSibling;
            if (icon) {
                icon.className = data.success ? 
                    'fas fa-check-circle mr-2 text-green-500' : 
                    'fas fa-exclamation-triangle mr-2 text-yellow-500';
            }
            
            // Auto-close after delay if successful
            if (data.success) {
                setTimeout(() => {
                    statusBar.classList.add('hidden');
                }, 5000);
            }
        })
        .catch(error => {
            console.error('Error exporting to CSV:', error);
            statusTitle.textContent = 'Export Failed';
            statusMessage.textContent = 'An error occurred while exporting articles.';
            
            // Change icon to error
            const icon = statusTitle.previousElementSibling;
            if (icon) {
                icon.className = 'fas fa-times-circle mr-2 text-red-500';
            }
        });
    }
}

// Function to clean up duplicates
function cleanupDuplicates() {
    // Update the sync status bar
    const statusBar = document.getElementById('updateStatusBar');
    const statusTitle = document.getElementById('updateStatusTitle');
    const statusMessage = document.getElementById('updateStatusMessage');
    
    if (statusBar && statusTitle && statusMessage) {
        statusBar.classList.remove('hidden');
        statusTitle.textContent = 'Cleaning Up Duplicates';
        statusMessage.textContent = 'Removing duplicate articles from database...';
        
        fetch('/api/storage/cleanup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            statusTitle.textContent = data.success ? 'Cleanup Complete' : 'Cleanup Warning';
            statusMessage.textContent = data.message;
            
            const icon = statusTitle.previousElementSibling;
            if (icon) {
                icon.className = data.success ? 
                    'fas fa-check-circle mr-2 text-green-500' : 
                    'fas fa-exclamation-triangle mr-2 text-yellow-500';
            }
            
            if (data.success) {
                setTimeout(() => {
                    statusBar.classList.add('hidden');
                }, 5000);
            }
        })
        .catch(error => {
            console.error('Error cleaning up duplicates:', error);
            statusTitle.textContent = 'Cleanup Failed';
            statusMessage.textContent = 'An error occurred while cleaning up duplicates.';
            
            const icon = statusTitle.previousElementSibling;
            if (icon) {
                icon.className = 'fas fa-times-circle mr-2 text-red-500';
            }
        });
    }
}

// Initialize storage admin functionality
document.addEventListener('DOMContentLoaded', initializeStorageAdmin);

// Update the loadMoreArticles function to handle errors better
async function loadMoreArticles() {
    if (isLoading || !hasMore) return;
    
    try {
        isLoading = true;
        loadingIndicator.classList.remove('hidden');
        
        // Get current URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        const nextPage = parseInt(urlParams.get('page') || '1') + 1;
        const searchQuery = urlParams.get('search_query') || '';
        const selectedTopic = urlParams.get('selected_topic') || '';
        const sortBy = urlParams.get('sort_by') || 'date';
        const sortOrder = urlParams.get('sort_order') || 'desc';
        
        // Build query parameters
        const queryParams = new URLSearchParams({
            page: nextPage.toString(),
            per_page: '9',
            sort_by: sortBy,
            sort_order: sortOrder
        });

        if (searchQuery) {
            queryParams.append('search_query', searchQuery);
        }
        if (selectedTopic) {
            queryParams.append('selected_topic', selectedTopic);
        }

        // Fetch new articles
        const response = await fetch(`/api/articles?${queryParams.toString()}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        if (!data.articles || data.articles.length === 0) {
            hasMore = false;
            loadingIndicator.classList.add('hidden');
            return;
        }
        
        // Create and append new articles
        const articlesGrid = document.querySelector('.articles-grid');
        if (!articlesGrid) {
            console.error('Articles grid not found');
            return;
        }
        
        const fragment = document.createDocumentFragment();
        data.articles.forEach(article => {
            const card = createArticleCard(article);
            if (card) {
                fragment.appendChild(card);
            }
        });
        
        // Update URL after successful fetch
        const url = new URL(window.location);
        url.searchParams.set('page', nextPage.toString());
        window.history.pushState({}, '', url);
        
        // Append all cards at once for better performance
        articlesGrid.appendChild(fragment);
        
    } catch (error) {
        console.error('Error loading more articles:', error);
        hasMore = false;
        
        // Show error toast
        const toast = document.createElement('div');
        toast.className = 'fixed bottom-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg shadow-lg z-50 flex items-center';
        toast.innerHTML = `
            <i class="fas fa-exclamation-circle mr-2"></i>
            <span>Error loading articles. Please try again.</span>
            <button class="ml-4 hover:text-gray-200" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 5000);
        
    } finally {
        // Add a small delay before hiding loading indicator for smoother transition
        setTimeout(() => {
            isLoading = false;
            loadingIndicator.classList.add('hidden');
        }, 300);
    }
}

// Initialize search form functionality
function initializeSearchForm() {
    const searchForm = document.querySelector('.search-form');
    const searchInput = document.getElementById('search_query');
    const clearInputBtn = document.getElementById('clearSearchInput');
    const topicSelect = document.getElementById('selected_topic');
    const sortBySelect = document.getElementById('sort_by');
    const sortOrderBtn = document.getElementById('sortOrderBtn');
    const clearSearchBtn = document.getElementById('clearSearch');
    const searchLoading = document.querySelector('.search-loading');
    const searchIcon = document.querySelector('.search-icon');
    const searchSuggestions = document.getElementById('searchSuggestions');
    const advancedSearchBtn = document.getElementById('toggleAdvancedSearch');
    const advancedSearchOptions = document.getElementById('advancedSearchOptions');
    const activeFilters = document.querySelector('.active-filters');

    // Initialize keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Press '/' to focus search
        if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
            e.preventDefault();
            searchInput?.focus();
        }
        // Press 'Escape' to clear search
        if (e.key === 'Escape' && document.activeElement === searchInput) {
            searchInput.value = '';
            searchInput.blur();
            updateClearButtonVisibility();
        }
    });

    // Toggle advanced search
    if (advancedSearchBtn && advancedSearchOptions) {
        advancedSearchBtn.addEventListener('click', () => {
            const isHidden = advancedSearchOptions.classList.contains('hidden');
            advancedSearchOptions.classList.toggle('hidden');
            advancedSearchBtn.setAttribute('aria-expanded', !isHidden);
            
            // Update button text
            const buttonText = advancedSearchBtn.querySelector('span');
            const buttonIcon = advancedSearchBtn.querySelector('i');
            if (buttonText && buttonIcon) {
                if (isHidden) {
                    buttonText.textContent = 'Simple Search';
                    buttonIcon.className = 'fas fa-minus mr-2';
                } else {
                    buttonText.textContent = 'Advanced Search';
                    buttonIcon.className = 'fas fa-sliders-h mr-2';
                }
            }
        });
    }

    // Initialize clear input button
    function updateClearButtonVisibility() {
        if (clearInputBtn) {
            clearInputBtn.classList.toggle('hidden', !searchInput?.value);
        }
    }

    if (searchInput && clearInputBtn) {
        searchInput.addEventListener('input', updateClearButtonVisibility);
        clearInputBtn.addEventListener('click', () => {
            searchInput.value = '';
            searchInput.focus();
            updateClearButtonVisibility();
        });
        // Initial state
        updateClearButtonVisibility();
    }

    // Debounced search suggestions
    const debouncedFetchSuggestions = debounce(async (query) => {
        if (!query || query.length < 2) {
            if (searchSuggestions) {
                searchSuggestions.classList.add('hidden');
            }
            return;
        }

        try {
            if (searchIcon) searchIcon.classList.add('hidden');
            if (searchLoading) searchLoading.classList.remove('hidden');

            const response = await fetch(`/api/search/suggestions?q=${encodeURIComponent(query)}`);
            if (!response.ok) {
                throw new Error('Failed to fetch suggestions');
            }

            const data = await response.json();
            
            if (searchSuggestions && data.suggestions && data.suggestions.length > 0) {
                searchSuggestions.innerHTML = data.suggestions.map(suggestion => `
                    <button class="w-full text-left px-4 py-2 hover:bg-gray-100 focus:bg-gray-100 focus:outline-none" 
                            onclick="selectSearchSuggestion('${escapeHtml(suggestion)}')"
                            role="option">
                        <i class="fas fa-search mr-2 text-gray-400"></i>
                        ${escapeHtml(suggestion)}
                    </button>
                `).join('');
                searchSuggestions.classList.remove('hidden');
            } else if (searchSuggestions) {
                searchSuggestions.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            if (searchSuggestions) {
                searchSuggestions.classList.add('hidden');
            }
        } finally {
            if (searchIcon) searchIcon.classList.remove('hidden');
            if (searchLoading) searchLoading.classList.add('hidden');
        }
    }, 300);

    // Initialize search input
    if (searchInput) {
        // Handle input changes for suggestions
        searchInput.addEventListener('input', (e) => {
            debouncedFetchSuggestions(e.target.value);
        });

        // Handle keyboard navigation in suggestions
        searchInput.addEventListener('keydown', (e) => {
            if (!searchSuggestions || searchSuggestions.classList.contains('hidden')) {
                return;
            }

            const suggestions = searchSuggestions.querySelectorAll('button');
            const currentIndex = Array.from(suggestions).findIndex(el => el === document.activeElement);

            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    if (currentIndex < 0) {
                        suggestions[0]?.focus();
                    } else {
                        suggestions[Math.min(currentIndex + 1, suggestions.length - 1)]?.focus();
                    }
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    if (currentIndex > 0) {
                        suggestions[currentIndex - 1]?.focus();
                    } else {
                        searchInput.focus();
                    }
                    break;
                case 'Escape':
                    e.preventDefault();
                    searchSuggestions.classList.add('hidden');
                    searchInput.focus();
                    break;
                case 'Enter':
                    if (document.activeElement !== searchInput) {
                        e.preventDefault();
                        document.activeElement.click();
                    }
                    break;
            }
        });

        // Close suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (searchSuggestions && !searchInput.contains(e.target) && !searchSuggestions.contains(e.target)) {
                searchSuggestions.classList.add('hidden');
            }
        });
    }

    // Initialize filters
    function initializeFilter(element, showLoading = true) {
        if (!element) return;
        
        element.addEventListener('change', () => {
            if (searchForm) {
                if (showLoading && searchLoading) {
                    searchLoading.classList.remove('hidden');
                    if (searchIcon) searchIcon.classList.add('hidden');
                }
                searchForm.submit();
            }
        });
    }

    // Initialize all filters
    initializeFilter(topicSelect);
    initializeFilter(document.getElementById('source_filter'));
    initializeFilter(document.getElementById('reading_time'));
    initializeFilter(sortBySelect);

    // Initialize date range inputs
    const dateFrom = document.querySelector('input[name="date_from"]');
    const dateTo = document.querySelector('input[name="date_to"]');

    [dateFrom, dateTo].forEach(dateInput => {
        if (dateInput) {
            dateInput.addEventListener('change', () => {
                if (searchForm) {
                    if (searchLoading) {
                        searchLoading.classList.remove('hidden');
                        if (searchIcon) searchIcon.classList.add('hidden');
                    }
                    searchForm.submit();
                }
            });
        }
    });

    // Initialize checkbox filters
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        initializeFilter(checkbox, true);
    });

    // Initialize sort order button
    if (sortOrderBtn) {
        sortOrderBtn.addEventListener('click', function() {
            const currentOrder = this.getAttribute('data-order');
            const newOrder = currentOrder === 'asc' ? 'desc' : 'asc';
            
            // Update button state
            this.setAttribute('data-order', newOrder);
            this.querySelector('i').className = `fas fa-sort-amount-${newOrder}`;
            
            // Update hidden input
            const sortOrderInput = document.getElementById('sort_order');
            if (sortOrderInput) {
                sortOrderInput.value = newOrder;
            }
            
            // Submit form
            if (searchForm) {
                if (searchLoading) {
                    searchLoading.classList.remove('hidden');
                    if (searchIcon) searchIcon.classList.add('hidden');
                }
                searchForm.submit();
            }
        });
    }

    // Initialize active filters
    if (activeFilters) {
        activeFilters.addEventListener('click', (e) => {
            const removeButton = e.target.closest('button[data-clear]');
            if (!removeButton) return;

            const filterType = removeButton.getAttribute('data-clear');
            
            switch (filterType) {
                case 'search_query':
                    if (searchInput) searchInput.value = '';
                    break;
                case 'selected_topic':
                    if (topicSelect) topicSelect.value = '';
                    break;
                case 'source':
                    const sourceSelect = document.getElementById('source_filter');
                    if (sourceSelect) sourceSelect.value = '';
                    break;
                case 'date':
                    if (dateFrom) dateFrom.value = '';
                    if (dateTo) dateTo.value = '';
                    break;
                case 'reading_time':
                    const readingTimeSelect = document.getElementById('reading_time');
                    if (readingTimeSelect) readingTimeSelect.value = '';
                    break;
            }

            if (searchForm) {
                if (searchLoading) {
                    searchLoading.classList.remove('hidden');
                    if (searchIcon) searchIcon.classList.add('hidden');
                }
                searchForm.submit();
            }
        });
    }

    // Initialize clear search
    if (clearSearchBtn) {
        clearSearchBtn.addEventListener('click', () => {
            // Reset all form inputs
            if (searchInput) searchInput.value = '';
            if (topicSelect) topicSelect.value = '';
            if (sortBySelect) sortBySelect.value = 'date';
            if (sortOrderBtn) {
                sortOrderBtn.setAttribute('data-order', 'desc');
                sortOrderBtn.querySelector('i').className = 'fas fa-sort-amount-desc';
            }
            
            // Reset advanced search options
            if (dateFrom) dateFrom.value = '';
            if (dateTo) dateTo.value = '';
            
            const sourceSelect = document.getElementById('source_filter');
            if (sourceSelect) sourceSelect.value = '';
            
            const readingTimeSelect = document.getElementById('reading_time');
            if (readingTimeSelect) readingTimeSelect.value = '';
            
            document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = false;
            });

            // Submit form
            if (searchForm) {
                if (searchLoading) {
                    searchLoading.classList.remove('hidden');
                    if (searchIcon) searchIcon.classList.add('hidden');
                }
                searchForm.submit();
            }
        });
    }

    // Initialize form submission
    if (searchForm) {
        searchForm.addEventListener('submit', () => {
            if (searchLoading) {
                searchLoading.classList.remove('hidden');
                if (searchIcon) searchIcon.classList.add('hidden');
            }
            // Hide suggestions when form is submitted
            if (searchSuggestions) {
                searchSuggestions.classList.add('hidden');
            }
        });
    }
}

// Initialize all functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeSearchForm();
    initializeStorageAdmin();
    initializeTopicDistribution();
    
    // Initialize infinite scroll
    window.addEventListener('scroll', checkScrollPosition);
});

// KumbyAI Chat Functionality
document.addEventListener('DOMContentLoaded', function() {
    const chatButton = document.getElementById('kumbyaiChatButton');
    // If chat button doesn't exist, the chat functionality is not available on this page
    if (!chatButton) return;

    const chatModal = document.getElementById('kumbyaiChatModal');
    const chatClose = document.getElementById('kumbyaiChatClose');
    const chatInput = document.getElementById('kumbyaiChatInput');
    const chatSend = document.getElementById('kumbyaiChatSend');
    const chatBody = document.getElementById('kumbyaiChatBody');

    // Ensure all required elements exist
    if (!chatModal || !chatClose || !chatInput || !chatSend || !chatBody) {
        console.warn('Some chat elements are missing. Chat functionality will be disabled.');
        return;
    }

    // Toggle chat modal
    chatButton.addEventListener('click', () => {
        chatModal.classList.toggle('active');
        chatInput.focus();
    });

    chatClose.addEventListener('click', () => {
        chatModal.classList.remove('active');
    });

    // Handle chat input
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    chatSend.addEventListener('click', sendMessage);

    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message to chat
        appendMessage('user', message);
        chatInput.value = '';

        // Send to backend and get response
        fetch('/api/kumbyai/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage('bot', data.response);
        })
        .catch(error => {
            console.error('Error:', error);
            appendMessage('bot', 'Sorry, I encountered an error. Please try again.');
        });
    }

    function appendMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        // Safely parse markdown content
        let parsedContent;
        try {
            parsedContent = window.marked ? marked.parse(content) : content;
        } catch (error) {
            console.warn('Error parsing markdown:', error);
            parsedContent = content;
        }
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <span class="message-author">${type === 'user' ? 'You' : 'KumbyAI'}</span>
                    <span class="message-time">${timestamp}</span>
                </div>
                <div class="message-text">${parsedContent}</div>
            </div>
        `;
        
        chatBody.appendChild(messageDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    // Add initial welcome message
    appendMessage('bot', 'Hello! I\'m KumbyAI, your pharmaceutical news assistant. How can I help you today?');
});

// Admin Functionality
document.addEventListener('DOMContentLoaded', function() {
    const adminDropdownBtn = document.getElementById('adminDropdownBtn');
    const adminDropdownMenu = document.getElementById('adminDropdownMenu');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileMenu = document.getElementById('mobileMenu');

    // Toggle admin dropdown
    if (adminDropdownBtn && adminDropdownMenu) {
        adminDropdownBtn.addEventListener('click', () => {
            adminDropdownMenu.classList.toggle('hidden');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!adminDropdownBtn.contains(e.target) && !adminDropdownMenu.contains(e.target)) {
                adminDropdownMenu.classList.add('hidden');
            }
        });
    }

    // Toggle mobile menu
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }

    // Admin functions
    const adminFunctions = {
        'exportToCsvBtn': () => handleExportToCsv(),
        'forceSyncBtn': () => handleForceSync(),
        'cleanupDuplicatesBtn': () => handleCleanupDuplicates(),
        'mobileExportBtn': () => handleExportToCsv(),
        'mobileForceSyncBtn': () => handleForceSync(),
        'mobileCleanupBtn': () => handleCleanupDuplicates()
    };

    // Add click handlers for admin functions
    Object.entries(adminFunctions).forEach(([id, handler]) => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('click', handler);
        }
    });
});

// Admin function handlers
async function handleExportToCsv() {
    try {
        const response = await fetch('/api/admin/export-csv');
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pharmaceutical-news-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
    } catch (error) {
        console.error('Export error:', error);
        alert('Failed to export data. Please try again.');
    }
}

async function handleForceSync() {
    try {
        const response = await fetch('/api/admin/force-sync', { method: 'POST' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        alert('Sync completed successfully!');
    } catch (error) {
        console.error('Sync error:', error);
        alert('Failed to sync data. Please try again.');
    }
}

async function handleCleanupDuplicates() {
    if (!confirm('Are you sure you want to clean up duplicate articles? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch('/api/admin/cleanup-duplicates', { method: 'POST' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        alert(`Cleanup completed! Removed ${data.removed_count} duplicate articles.`);
    } catch (error) {
        console.error('Cleanup error:', error);
        alert('Failed to clean up duplicates. Please try again.');
    }
} 