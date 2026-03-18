(function() {
    // Map column show/hide
    function initMap() {
        const el = document.getElementById('map-col');
        if (!el) { setTimeout(initMap, 300); return; }
        new MutationObserver(function() {
            el.classList.toggle('map-active', !!el.querySelector('iframe'));
        }).observe(el, { childList: true, subtree: true });
    }
    setTimeout(initMap, 500);

    // Centered input: CSS defaults to centered; JS adds chat-active when messages exist
    function initChatCenter() {
        const chatCol = document.getElementById('chat-col-inner');
        if (!chatCol) { setTimeout(initChatCenter, 300); return; }

        function setActive(active) {
            chatCol.classList.toggle('chat-active', active);
            document.body.classList.toggle('chat-active', active);
        }

        function checkState() {
            const chatbot = document.getElementById('main-chatbot');
            if (!chatbot) return;
            // Gradio 4.x wraps messages in [role="log"]
            const log = chatbot.querySelector('[role="log"]');
            const hasMsgs = log ? log.children.length > 0 : false;
            setActive(hasMsgs);
        }

        // Observe the stable column — survives Gradio's DOM rebuilds on clear/reset
        new MutationObserver(checkState).observe(chatCol, { childList: true, subtree: true });
    }
    setTimeout(initChatCenter, 300);
})();
