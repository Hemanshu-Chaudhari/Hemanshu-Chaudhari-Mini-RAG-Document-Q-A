document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("pdfFileInput");
    const uploadProgress = document.getElementById("uploadProgress");
    const uploadStatusText = document.getElementById("uploadStatusText");
    const totalChunksVal = document.getElementById("totalChunksVal");
    const documentsList = document.getElementById("documentsList");
    const refreshStatsBtn = document.getElementById("refreshStatsBtn");
    const clearDbBtn = document.getElementById("clearDbBtn");
    const resetChatBtn = document.getElementById("resetChatBtn");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const chatMessages = document.getElementById("chatMessages");
    const toggleMemory = document.getElementById("toggleMemory");
    const toggleReranker = document.getElementById("toggleReranker");
    const keyStatusBadge = document.getElementById("keyStatusBadge");
    const apiKeyInput = document.getElementById("apiKeyInput");
    const saveKeyBtn = document.getElementById("saveKeyBtn");

    // Check API Key status and initial collection stats
    checkApiKeyConfig();
    loadStats();

    // Check API Key Status
    async function checkApiKeyConfig() {
        try {
            const resp = await fetch("/api/config");
            const data = await resp.json();
            if (data.has_api_key) {
                keyStatusBadge.textContent = "Ready";
                keyStatusBadge.className = "status-badge badge-ok";
                apiKeyInput.placeholder = "API Key Active (Re-enter to update)";
            } else {
                keyStatusBadge.textContent = "Missing";
                keyStatusBadge.className = "status-badge badge-missing";
                apiKeyInput.placeholder = "Paste AI Studio Key...";
            }
        } catch (e) {
            console.error("Failed to check config:", e);
        }
    }

    // Save API Key
    saveKeyBtn.addEventListener("click", async () => {
        const key = apiKeyInput.value.trim();
        if (!key) {
            alert("Please enter a valid Gemini API Key.");
            return;
        }

        saveKeyBtn.disabled = true;
        saveKeyBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';

        try {
            const resp = await fetch("/api/set-key", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: key })
            });
            const data = await resp.json();
            if (data.success) {
                keyStatusBadge.textContent = "Ready";
                keyStatusBadge.className = "status-badge badge-ok";
                apiKeyInput.value = "";
                apiKeyInput.placeholder = "API Key Active (Re-enter to update)";
                appendSystemNotification("🔑 **Gemini API Key configured successfully!** You can now upload PDFs and ask questions.");
                loadStats();
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (err) {
            alert(`Failed to save key: ${err.message}`);
        } finally {
            saveKeyBtn.disabled = false;
            saveKeyBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i>';
        }
    });


    // Auto-resize textarea
    userInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 120) + "px";
    });

    userInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Drag and drop handlers
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Upload and Ingest File
    async function handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            alert("Please upload a valid PDF document.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        uploadProgress.style.display = "block";
        uploadStatusText.textContent = `Extracting, chunking & indexing "${file.name}"...`;

        try {
            const resp = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();

            if (data.success) {
                uploadStatusText.textContent = `Indexed ${data.chunks_created} chunks successfully!`;
                setTimeout(() => {
                    uploadProgress.style.display = "none";
                }, 3000);
                updateStatsUI(data.stats);
                appendSystemNotification(`📄 Document **${data.filename}** ingested: ${data.chunks_created} semantic chunks added.`);
            } else {
                alert(`Ingestion failed: ${data.error}`);
                uploadProgress.style.display = "none";
            }
        } catch (err) {
            alert(`Network error: ${err.message}`);
            uploadProgress.style.display = "none";
        }
    }

    // Load Stats
    async function loadStats() {
        try {
            const resp = await fetch("/api/stats");
            const data = await resp.json();
            if (data.success) {
                updateStatsUI(data.stats);
            }
        } catch (err) {
            console.error("Failed to load stats:", err);
        }
    }

    function updateStatsUI(stats) {
        if (!stats) return;
        totalChunksVal.textContent = stats.total_chunks || 0;
        documentsList.innerHTML = "";
        if (stats.documents && stats.documents.length > 0) {
            stats.documents.forEach((doc) => {
                const li = document.createElement("li");
                li.innerHTML = `<i class="fa-solid fa-file-pdf" style="color: #ef4444;"></i> ${doc}`;
                documentsList.appendChild(li);
            });
        } else {
            documentsList.innerHTML = '<li class="empty-hint">No documents indexed yet</li>';
        }
    }

    refreshStatsBtn.addEventListener("click", loadStats);

    // Clear Collection
    clearDbBtn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear the vector database? All indexed chunks will be deleted.")) {
            return;
        }
        try {
            const resp = await fetch("/api/clear-collection", { method: "POST" });
            const data = await resp.json();
            if (data.success) {
                loadStats();
                appendSystemNotification("🧹 Vector database cleared.");
            }
        } catch (err) {
            alert(err.message);
        }
    });

    // Reset Chat Memory
    resetChatBtn.addEventListener("click", async () => {
        try {
            const resp = await fetch("/api/reset-memory", { method: "POST" });
            const data = await resp.json();
            if (data.success) {
                appendSystemNotification("🔄 Conversation memory reset. Next question will start a fresh context.");
            }
        } catch (err) {
            alert(err.message);
        }
    });

    // Chat Submission
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // Append user message
        appendMessage("user", text);
        userInput.value = "";
        userInput.style.height = "auto";

        // Show typing indicator
        const typingIndicator = appendTypingIndicator();

        try {
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    use_memory: toggleMemory.checked,
                    use_reranking: toggleReranker.checked
                })
            });
            const data = await resp.json();
            typingIndicator.remove();

            if (data.success) {
                appendMessage("assistant", data.answer, data.sources, data.standalone_query, text);
            } else {
                appendMessage("assistant", `⚠️ **Error:** ${data.error}`);
            }
        } catch (err) {
            typingIndicator.remove();
            appendMessage("assistant", `⚠️ **Network Error:** ${err.message}`);
        }
    });

    function appendMessage(role, text, sources = null, standaloneQuery = null, originalQuery = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}-message`;

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.innerHTML = role === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

        const content = document.createElement("div");
        content.className = "msg-content";

        // If assistant and query was reformulated by conversational memory
        if (role === "assistant" && standaloneQuery && originalQuery && standaloneQuery.toLowerCase() !== originalQuery.toLowerCase()) {
            const badge = document.createElement("div");
            badge.className = "query-reformulation-badge";
            badge.innerHTML = `<i class="fa-solid fa-magnifying-glass"></i> Contextual Search Query: <em>"${escapeHtml(standaloneQuery)}"</em>`;
            content.appendChild(badge);
        }

        // Render Markdown content
        const textContainer = document.createElement("div");
        textContainer.innerHTML = marked.parse(text);
        content.appendChild(textContainer);

        // Render citations/sources if available
        if (role === "assistant" && sources && sources.length > 0) {
            const sourcesPanel = document.createElement("div");
            sourcesPanel.className = "sources-panel";

            const toggle = document.createElement("button");
            toggle.className = "sources-toggle";
            toggle.innerHTML = `<i class="fa-solid fa-bookmark"></i> Sources & Page Citations (${sources.length})`;
            
            const list = document.createElement("div");
            list.className = "citations-list";

            sources.forEach((s) => {
                const chip = document.createElement("div");
                chip.className = "citation-chip";
                const scoreBadge = s.rerank_score !== null 
                    ? `<span class="citation-score">Score: ${s.rerank_score.toFixed(1)}/10</span>` 
                    : `<span class="citation-score">Sim: ${(s.similarity * 100).toFixed(0)}%</span>`;
                chip.innerHTML = `<i class="fa-solid fa-file-lines"></i> ${escapeHtml(s.source)} <strong>[Page ${s.page}]</strong> ${scoreBadge}`;
                list.appendChild(chip);
            });

            sourcesPanel.appendChild(toggle);
            sourcesPanel.appendChild(list);
            content.appendChild(sourcesPanel);
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendTypingIndicator() {
        const div = document.createElement("div");
        div.className = "message assistant-message";
        div.innerHTML = `
            <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="msg-content" style="color: var(--text-muted);">
                <i class="fa-solid fa-circle-notch fa-spin"></i> Retrieving context and generating answer...
            </div>
        `;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return div;
    }

    function appendSystemNotification(mdText) {
        const div = document.createElement("div");
        div.className = "message";
        div.style.alignSelf = "center";
        div.style.maxWidth = "90%";
        div.innerHTML = `
            <div class="msg-content" style="background: rgba(56, 189, 248, 0.08); border-color: rgba(56, 189, 248, 0.2); font-size: 13px; text-align: center;">
                ${marked.parse(mdText)}
            </div>
        `;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function escapeHtml(str) {
        const p = document.createElement("p");
        p.textContent = str;
        return p.innerHTML;
    }
});
