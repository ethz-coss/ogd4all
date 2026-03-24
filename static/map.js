(function () {

    /* ── 1. Desktop: map column slide-in ── */
    function initMapCol() {
        var el = document.getElementById('map-col');
        if (!el) { setTimeout(initMapCol, 300); return; }
        new MutationObserver(function () {
            var hasIframe = !!el.querySelector('iframe');
            el.classList.toggle('map-active', hasIframe);
            setMapReady(hasIframe);
        }).observe(el, { childList: true, subtree: true });
    }
    setTimeout(initMapCol, 500);

    /* ── 2. Chat active state (centered → bottom input) ── */
    function initChatCenter() {
        var chatCol = document.getElementById('chat-col-inner');
        if (!chatCol) { setTimeout(initChatCenter, 300); return; }

        function check() {
            var chatbot = document.getElementById('main-chatbot');
            if (!chatbot) return;
            var log = chatbot.querySelector('[role="log"]');
            var active = log ? log.children.length > 0 : false;
            chatCol.classList.toggle('chat-active', active);
            document.body.classList.toggle('chat-active', active);
        }

        // Observe the stable column — survives Gradio's DOM rebuilds on clear/reset
        new MutationObserver(check).observe(chatCol, { childList: true, subtree: true });
    }
    setTimeout(initChatCenter, 300);

    /* ── 3. Mobile: in-chat map button & fullscreen overlay ── */
    var overlay = null;
    var mapChatBtn = null;
    var mapReady = false;
    var reinjecting = false;
    var imgLightbox = null;

    function setMapReady(ready) {
        mapReady = ready;
        if (ready) {
            ensureMapBtnLast();
            // If overlay is open and map updated, refresh its content
            if (overlay && overlay.classList.contains('active')) {
                loadMapIntoOverlay();
            }
        } else {
            if (mapChatBtn) { mapChatBtn.remove(); mapChatBtn = null; }
        }
    }

    /* Keep the map button as the last child of [role="log"] */
    function ensureMapBtnLast() {
        if (reinjecting || !mapReady) return;
        var chatbot = document.getElementById('main-chatbot');
        var log = chatbot && chatbot.querySelector('[role="log"]');
        if (!log) return;

        if (!mapChatBtn) {
            mapChatBtn = buildMapChatBtn();
        }

        if (log.lastElementChild !== mapChatBtn) {
            reinjecting = true;
            log.appendChild(mapChatBtn);
            // Reset flag after microtasks + one macrotask so the observer
            // callback triggered by appendChild is already processed
            setTimeout(function () { reinjecting = false; }, 0);
        }
    }

    function buildMapChatBtn() {
        var wrap = document.createElement('div');
        wrap.id = 'map-chat-btn';
        var btn = document.createElement('button');
        btn.setAttribute('aria-label', 'Karte anzeigen');
        btn.innerHTML =
            '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
            '<path d="M9 4L3 7v13l6-3 6 3 6-3V4l-6 3-6-3z"/>' +
            '<line x1="9" y1="4" x2="9" y2="17"/><line x1="15" y1="7" x2="15" y2="20"/>' +
            '</svg>' +
            '<span>Karte anzeigen</span>';
        btn.addEventListener('click', openMapOverlay);
        wrap.appendChild(btn);
        return wrap;
    }

    /* Observer on chatbot: keeps map button last during streaming */
    function initChatBotObserver() {
        var chatbot = document.getElementById('main-chatbot');
        if (!chatbot) { setTimeout(initChatBotObserver, 400); return; }
        new MutationObserver(function () {
            if (!reinjecting) ensureMapBtnLast();
        }).observe(chatbot, { childList: true, subtree: true });
    }
    setTimeout(initChatBotObserver, 600);

    /* ── 4. Fullscreen overlay ── */
    function buildOverlay() {
        var chatCol = document.getElementById('chat-col-inner');
        if (!chatCol) { setTimeout(buildOverlay, 400); return; }

        overlay = document.createElement('div');
        overlay.id = 'map-overlay';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.setAttribute('aria-label', 'Kartenansicht');
        overlay.innerHTML =
            '<div id="map-overlay-bar">' +
            '<span class="map-overlay-title">Karte</span>' +
            '<button id="map-overlay-close" aria-label="Karte schlie\u00dfen">' +
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">' +
            '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>' +
            '</svg></button></div>' +
            '<div id="map-overlay-content"></div>';
        document.body.appendChild(overlay);

        overlay.querySelector('#map-overlay-close').addEventListener('click', closeMapOverlay);
    }
    setTimeout(buildOverlay, 600);

    /* ── 5. Image lightbox for chatbot charts ── */
    function buildImageLightbox() {
        var chatbot = document.getElementById('main-chatbot');
        if (!chatbot) { setTimeout(buildImageLightbox, 400); return; }

        var lb = document.createElement('div');
        lb.id = 'img-lightbox';
        lb.setAttribute('role', 'dialog');
        lb.setAttribute('aria-modal', 'true');
        lb.setAttribute('aria-label', 'Full-size chart');
        var lbImg = document.createElement('img');
        lbImg.setAttribute('alt', '');
        lb.appendChild(lbImg);
        document.body.appendChild(lb);
        imgLightbox = lb;

        lb.addEventListener('click', function () {
            lb.classList.remove('active');
        });
        lbImg.addEventListener('click', function (e) {
            e.stopPropagation();
        });

        // Intercept clicks anywhere in .image-container (capture phase = before Gradio's broken handler)
        chatbot.addEventListener('click', function (e) {
            var container = e.target.closest('.image-container');
            if (!container) return;
            var img = container.querySelector('.image-frame img');
            if (!img || !img.src) return;
            lbImg.src = img.src;
            lb.classList.add('active');
            e.stopPropagation();
        }, true);
    }
    setTimeout(buildImageLightbox, 700);

    /* ── 6. Info modal & dark-mode toggle (mobile footer) ── */
    function initFooterBtns() {
        var infoBtn  = document.getElementById('footer-info-btn');
        var backdrop = document.getElementById('info-modal-backdrop');

        if (!infoBtn || !backdrop) { setTimeout(initFooterBtns, 400); return; }

        infoBtn.addEventListener('click', function () {
            backdrop.classList.add('active');
            document.documentElement.style.overflow = 'hidden';
        });

        function closeInfoModal() {
            backdrop.classList.remove('active');
            document.documentElement.style.overflow = '';
        }

        backdrop.addEventListener('click', function (e) {
            if (e.target === backdrop) closeInfoModal();
        });

        document.addEventListener('keydown', function (e) {
            if (e.key !== 'Escape') return;
            if (imgLightbox && imgLightbox.classList.contains('active')) { imgLightbox.classList.remove('active'); return; }
            if (backdrop.classList.contains('active')) closeInfoModal();
            else if (overlay && overlay.classList.contains('active')) closeMapOverlay();
        });
    }
    setTimeout(initFooterBtns, 700);

    function loadMapIntoOverlay() {
        var mapCol = document.getElementById('map-col');
        var srcIframe = mapCol && mapCol.querySelector('iframe');
        if (!srcIframe) return;
        var content = document.getElementById('map-overlay-content');
        if (!content) return;

        content.innerHTML = '';
        var iframe = document.createElement('iframe');
        var srcdoc = srcIframe.srcdoc;
        if (srcdoc) {
            iframe.srcdoc = srcdoc;
        } else if (srcIframe.src) {
            iframe.src = srcIframe.src;
        }
        var sandbox = srcIframe.getAttribute('sandbox');
        if (sandbox) iframe.setAttribute('sandbox', sandbox);
        iframe.style.cssText = 'width:100%;height:100%;border:none;display:block;';
        content.appendChild(iframe);
    }

    function openMapOverlay() {
        if (!overlay) return;
        loadMapIntoOverlay();
        overlay.classList.add('active');
        document.documentElement.style.overflow = 'hidden';
    }

    function closeMapOverlay() {
        if (!overlay) return;
        overlay.classList.remove('active');
        document.documentElement.style.overflow = '';
        setTimeout(function () {
            if (!overlay.classList.contains('active')) {
                var content = document.getElementById('map-overlay-content');
                if (content) content.innerHTML = '';
            }
        }, 380);
    }

})();
