document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const fileList = document.getElementById('fileList');
    const processBtn = document.getElementById('processBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const clearChatBtn = document.getElementById('clearChat');
    const apiKeyInput = document.getElementById('apiKey');

    let uploadedFiles = [];

    // File Handling
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        for (const file of files) {
            if (file.type === 'application/pdf') {
                uploadedFiles.push(file);
            }
        }
        updateFileList();
        processBtn.disabled = uploadedFiles.length === 0;
    }

    function updateFileList() {
        fileList.innerHTML = '';
        uploadedFiles.forEach((file, index) => {
            const div = document.createElement('div');
            div.className = 'file-item';
            div.innerHTML = `<i class="fa-solid fa-file-pdf"></i> ${file.name} <i class="fa-solid fa-times" onclick="removeFile(${index})" style="cursor:pointer; margin-left:auto; color: #ef4444;"></i>`;
            fileList.appendChild(div);
        });
    }

    window.removeFile = (index) => {
        uploadedFiles.splice(index, 1);
        updateFileList();
        processBtn.disabled = uploadedFiles.length === 0;
    }

    // Process Documents
    processBtn.addEventListener('click', async () => {
        const formData = new FormData();
        uploadedFiles.forEach(file => {
            formData.append('files', file);
        });
        
        const key = apiKeyInput.value.trim();
        if (key) {
            formData.append('api_key', key);
        }

        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        uploadStatus.textContent = '';
        uploadStatus.className = '';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                uploadStatus.textContent = "Documents processed successfully!";
                uploadStatus.className = "success";
                uploadStatus.style.color = "var(--success-color)";
            } else {
                throw new Error(data.error || "Upload failed");
            }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.className = "error";
            uploadStatus.style.color = "var(--error-color)";
        } finally {
            processBtn.innerHTML = '<i class="fa-solid fa-gears"></i> Process Documents';
            processBtn.disabled = false;
        }
    });

    // Chat Logic
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = userInput.value.trim();
        if (!question) return;

        // User Message
        appendMessage('user', question);
        userInput.value = '';

        // Bot Loading
        const loadingId = appendMessage('bot', '<i class="fa-solid fa-circle-notch fa-spin"></i> Thinking...');

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    api_key: apiKeyInput.value.trim()
                })
            });
            
            const data = await response.json();
            
            removeMessage(loadingId);

            if (response.ok) {
                let content = marked.parse(data.answer);
                
                // Append Sources
                if (data.sources && data.sources.length > 0) {
                    content += `<div class="sources-container"><strong>Sources:</strong>`;
                    data.sources.forEach((src, i) => {
                        content += `
                        <div class="source-item">
                            <div class="source-title">Source ${i+1} (Page ${src.page}, ${src.source})</div>
                            <div class="source-text">"${src.content}"</div>
                        </div>`;
                    });
                    content += `</div>`;
                }
                
                appendMessage('bot', content);
            } else {
                appendMessage('bot', `Error: ${data.error || "Failed to get answer"}`);
            }

        } catch (error) {
            removeMessage(loadingId);
            appendMessage('bot', `Error: ${error.message}`);
        }
    });

    function appendMessage(role, htmlContent) {
        const id = 'msg-' + Date.now();
        const div = document.createElement('div');
        div.id = id;
        div.className = `message ${role === 'user' ? 'user-message' : 'system-message'}`;
        
        const avatar = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        div.innerHTML = `
            <div class="avatar">${avatar}</div>
            <div class="content">${htmlContent}</div>
        `;
        
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    clearChatBtn.addEventListener('click', () => {
        chatContainer.innerHTML = '';
        appendMessage('bot', '<p>Chat cleared. How can I help you with your documents?</p>');
    });
});
