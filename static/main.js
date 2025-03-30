document.addEventListener("DOMContentLoaded", () => {
    const messageInput = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");
    const chatHistory = document.getElementById("chat-history");
    const loadingIndicator = document.getElementById("loading-indicator");
    const reconnectButton = document.getElementById("reconnect-button");
    const sessionInfo = document.getElementById("session-info");

    const userId = localStorage.getItem("userId") || "Anonim"; // Kullanıcı adı
    let sessionId = localStorage.getItem("sessionId") || null; // Oturum ID

    // Yeni bir oturum oluştur
    async function createSession() {
        try {
            setLoading(true);
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: "start_session", user_id: userId })
            });

            if (!response.ok) {
                throw new Error(`Oturum oluşturulamadı. Durum kodu: ${response.status}`);
            }

            const data = await response.json();
            if (data.status === "success") {
                sessionId = data.session_id;
                localStorage.setItem("sessionId", sessionId);
                updateSessionInfo(sessionId);

                const welcomeMessage = userId !== "Anonim" 
                    ? `Merhaba, ${userId}! Size nasıl yardımcı olabilirim?` 
                    : "Merhaba! Size nasıl yardımcı olabilirim?";
                appendMessage("bot", welcomeMessage);
            } else {
                throw new Error(data.message || "Bilinmeyen bir hata oluştu.");
            }
        } catch (error) {
            console.error("Oturum oluşturulurken hata oluştu:", error);
            appendMessage("bot", "Oturum başlatılamadı. Lütfen yeniden deneyin.");
            showReconnectOption();
        } finally {
            setLoading(false);
        }
    }

    // Sayfa yüklendiğinde oturumu başlat
    if (!sessionId) {
        createSession();
    } else {
        updateSessionInfo(sessionId);
        loadChatHistory();
    }

    // Mesaj gönderme
    sendButton.addEventListener("click", async () => {
        const message = messageInput.value.trim();
        if (!message) return;

        appendMessage("user", message); // Kullanıcı mesajını arayüze ekle
        await sendMessageToServer(message);
        messageInput.value = ""; // Girdi alanını temizle
    });

    // Enter tuşuna basıldığında mesaj gönderme
    messageInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            sendButton.click();
        }
    });

    // Yeniden bağlanma
    reconnectButton.addEventListener("click", () => {
        reconnectButton.style.display = "none";
        sessionId = null;
        localStorage.removeItem("sessionId");
        createSession();
    });

    // Sunucuya mesaj gönder
    async function sendMessageToServer(message) {
        try {
            setLoading(true);
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, message })
            });

            if (!response.ok) {
                throw new Error(`Mesaj gönderilemedi. Durum kodu: ${response.status}`);
            }

            const data = await response.json();
            if (data.status === "success") {
                appendMessage("bot", data.response);
            } else {
                throw new Error(data.message || "Bilinmeyen bir hata oluştu.");
            }
        } catch (error) {
            console.error("Mesaj gönderilirken hata oluştu:", error);
            appendMessage("bot", "Mesaj gönderilemedi. Lütfen tekrar deneyin.");
        } finally {
            setLoading(false);
        }
    }

    // Sohbet geçmişini yükle
    async function loadChatHistory() {
        try {
            const response = await fetch(`/api/get_chat_history?session_id=${sessionId}`);
            if (!response.ok) {
                throw new Error(`Geçmiş yüklenemedi. Durum kodu: ${response.status}`);
            }

            const data = await response.json();
            if (data.status === "success" && data.chat_history) {
                data.chat_history.forEach(chat => {
                    const sender = chat.Soru ? "user" : "bot";
                    const message = chat.Soru || chat.Cevap;
                    appendMessage(sender, message);
                });
            } else {
                appendMessage("bot", "Geçmiş yüklenemedi.");
            }
        } catch (error) {
            console.error("Sohbet geçmişi yüklenirken hata oluştu:", error);
            appendMessage("bot", "Geçmiş yüklenemedi. Lütfen tekrar deneyin.");
        }
    }

    // Mesajı ekrana ekleme
    function appendMessage(sender, text) {
        const messageElement = document.createElement("div");
        messageElement.className = `message ${sender}`;
        messageElement.textContent = text;
        chatHistory.appendChild(messageElement);
        chatHistory.scrollTop = chatHistory.scrollHeight; // Otomatik kaydırma
    }

    // Yükleme göstergesi
    function setLoading(isLoading) {
        if (loadingIndicator) {
            loadingIndicator.style.display = isLoading ? "block" : "none";
        }
        messageInput.disabled = isLoading;
        sendButton.disabled = isLoading;
    }

    // Oturum bilgisini güncelle
    function updateSessionInfo(sessionId) {
        if (sessionInfo) {
            sessionInfo.textContent = `Oturum ID: ${sessionId}`;
            sessionInfo.style.display = "block";
        }
    }

    // Yeniden bağlanma seçeneğini göster
    function showReconnectOption() {
        if (reconnectButton) {
            reconnectButton.style.display = "block";
        }
    }

    // Hata yakalama
    window.addEventListener("error", (event) => {
        console.error("Hata:", event.error || event.message);
        appendMessage("bot", "Bir hata oluştu. Lütfen tekrar deneyin.");
    });

    window.addEventListener("unhandledrejection", (event) => {
        console.error("Yakalanmamış hata:", event.reason);
        appendMessage("bot", "Sistemsel bir hata oluştu. Lütfen tekrar deneyin.");
    });
});
