// =====================================================
// Simulation 4 — Latent Space Explorer
// =====================================================

(function () {
    'use strict';

    // We use a simple generative model that produces digit-like patterns
    // by blending pre-made templates based on latent vector values.

    const IMG_SIZE = 64;

    // --- Digit templates (simplified 8x8 patterns scaled to 64x64) ---
    const DIGIT_PATTERNS = [
        // 0
        [0,0,1,1,1,1,0,0, 0,1,1,0,0,1,1,0, 1,1,0,0,0,0,1,1, 1,1,0,0,0,0,1,1,
         1,1,0,0,0,0,1,1, 1,1,0,0,0,0,1,1, 0,1,1,0,0,1,1,0, 0,0,1,1,1,1,0,0],
        // 1
        [0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,1,0,1,1,0,0,0, 0,0,0,1,1,0,0,0,
         0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,1,1,1,1,1,1,0],
        // 2
        [0,0,1,1,1,1,0,0, 0,1,1,0,0,1,1,0, 0,0,0,0,0,1,1,0, 0,0,0,0,1,1,0,0,
         0,0,0,1,1,0,0,0, 0,0,1,1,0,0,0,0, 0,1,1,0,0,0,0,0, 0,1,1,1,1,1,1,0],
        // 3
        [0,0,1,1,1,1,0,0, 0,1,0,0,0,1,1,0, 0,0,0,0,0,1,1,0, 0,0,0,1,1,1,0,0,
         0,0,0,0,0,1,1,0, 0,0,0,0,0,1,1,0, 0,1,0,0,0,1,1,0, 0,0,1,1,1,1,0,0],
        // 4
        [0,0,0,0,1,1,0,0, 0,0,0,1,1,1,0,0, 0,0,1,0,1,1,0,0, 0,1,0,0,1,1,0,0,
         1,1,1,1,1,1,1,0, 0,0,0,0,1,1,0,0, 0,0,0,0,1,1,0,0, 0,0,0,0,1,1,0,0],
        // 5
        [0,1,1,1,1,1,1,0, 0,1,1,0,0,0,0,0, 0,1,1,0,0,0,0,0, 0,1,1,1,1,1,0,0,
         0,0,0,0,0,1,1,0, 0,0,0,0,0,1,1,0, 0,1,0,0,0,1,1,0, 0,0,1,1,1,1,0,0],
        // 6
        [0,0,1,1,1,1,0,0, 0,1,1,0,0,0,0,0, 1,1,0,0,0,0,0,0, 1,1,1,1,1,1,0,0,
         1,1,0,0,0,1,1,0, 1,1,0,0,0,1,1,0, 0,1,1,0,0,1,1,0, 0,0,1,1,1,1,0,0],
        // 7
        [0,1,1,1,1,1,1,0, 0,0,0,0,0,1,1,0, 0,0,0,0,1,1,0,0, 0,0,0,0,1,1,0,0,
         0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,1,1,0,0,0,0, 0,0,1,1,0,0,0,0],
        // 8
        [0,0,1,1,1,1,0,0, 0,1,1,0,0,1,1,0, 0,1,1,0,0,1,1,0, 0,0,1,1,1,1,0,0,
         0,1,1,0,0,1,1,0, 0,1,1,0,0,1,1,0, 0,1,1,0,0,1,1,0, 0,0,1,1,1,1,0,0],
        // 9
        [0,0,1,1,1,1,0,0, 0,1,1,0,0,1,1,0, 0,1,1,0,0,1,1,0, 0,0,1,1,1,1,1,0,
         0,0,0,0,0,1,1,0, 0,0,0,0,0,1,1,0, 0,0,0,0,1,1,0,0, 0,0,1,1,1,0,0,0]
    ];

    let mainCanvas, mainCtx;
    let interpContainer;
    let z1Slider, z2Slider, z3Slider;
    let z1Val, z2Val, z3Val;
    let dpr = 1;

    function sizeMainCanvas() {
        const container = mainCanvas.parentElement;
        const rect = container.getBoundingClientRect();
        dpr = window.devicePixelRatio || 1;
        const size = Math.min(rect.width, rect.height, 280);
        mainCanvas.width = size * dpr;
        mainCanvas.height = size * dpr;
        mainCanvas.style.width = size + 'px';
        mainCanvas.style.height = size + 'px';
        mainCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function init() {
        mainCanvas = document.getElementById('sim4Canvas');
        mainCtx = mainCanvas.getContext('2d');

        interpContainer = document.getElementById('sim4InterpStrip');

        z1Slider = document.getElementById('sim4Z1');
        z2Slider = document.getElementById('sim4Z2');
        z3Slider = document.getElementById('sim4Z3');
        z1Val = document.getElementById('sim4Z1Val');
        z2Val = document.getElementById('sim4Z2Val');
        z3Val = document.getElementById('sim4Z3Val');

        const btnRandom = document.getElementById('sim4Random');
        const btnInterp = document.getElementById('sim4Interp');

        z1Slider.oninput = () => { z1Val.textContent = parseFloat(z1Slider.value).toFixed(2); render(); };
        z2Slider.oninput = () => { z2Val.textContent = parseFloat(z2Slider.value).toFixed(2); render(); };
        z3Slider.oninput = () => { z3Val.textContent = parseFloat(z3Slider.value).toFixed(2); render(); };

        btnRandom.addEventListener('click', randomize);
        btnInterp.addEventListener('click', interpolate);

        requestAnimationFrame(() => { sizeMainCanvas(); render(); });
        window.addEventListener('resize', () => { sizeMainCanvas(); render(); });
    }

    function getLatentVector() {
        return [
            parseFloat(z1Slider.value),
            parseFloat(z2Slider.value),
            parseFloat(z3Slider.value)
        ];
    }

    function generateFromLatent(z) {
        // Map latent vector to a blend of digit patterns
        // z[0] selects primary digit (0-9)
        // z[1] controls morphing to neighboring digit
        // z[2] controls noise/variation

        const digitIdx = ((z[0] + 1) / 2) * 9; // Map [-1,1] → [0,9]
        const d1 = Math.floor(Math.max(0, Math.min(9, digitIdx)));
        const d2 = Math.min(9, d1 + 1);
        const blend = digitIdx - d1;

        const morphFactor = (z[1] + 1) / 2; // 0 to 1
        const noiseLevel = (z[2] + 1) / 2 * 0.3; // 0 to 0.3

        const pattern = new Float32Array(64);
        const p1 = DIGIT_PATTERNS[d1];
        const p2 = DIGIT_PATTERNS[d2];

        for (let i = 0; i < 64; i++) {
            // Blend between two digit patterns
            let val = p1[i] * (1 - blend * morphFactor) + p2[i] * blend * morphFactor;
            // Add noise variation
            val += (Math.random() - 0.5) * noiseLevel;
            pattern[i] = Math.max(0, Math.min(1, val));
        }

        return pattern;
    }

    function renderPattern(ctx, pattern, x, y, size) {
        const cellSize = size / 8;
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const val = pattern[row * 8 + col];
                const brightness = Math.floor(val * 220 + 20);
                ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
                ctx.fillRect(x + col * cellSize, y + row * cellSize, cellSize, cellSize);
            }
        }
    }

    function render() {
        const z = getLatentVector();
        const pattern = generateFromLatent(z);

        const cssW = parseInt(mainCanvas.style.width) || mainCanvas.width / dpr;
        const cssH = parseInt(mainCanvas.style.height) || mainCanvas.height / dpr;

        mainCtx.fillStyle = '#111';
        mainCtx.fillRect(0, 0, cssW, cssH);

        renderPattern(mainCtx, pattern, 0, 0, Math.min(cssW, cssH));

        // Label
        const digitIdx = ((z[0] + 1) / 2) * 9;
        const d1 = Math.floor(Math.max(0, Math.min(9, digitIdx)));
        const d2 = Math.min(9, d1 + 1);
        const blend = (digitIdx - d1).toFixed(2);

        const labelEl = document.getElementById('sim4DigitLabel');
        if (labelEl) {
            labelEl.textContent = `Digit ${d1}` + (d1 !== d2 ? ` → ${d2} (${(blend * 100).toFixed(0)}%)` : '');
        }
    }

    function randomize() {
        z1Slider.value = (Math.random() * 2 - 1).toFixed(2);
        z2Slider.value = (Math.random() * 2 - 1).toFixed(2);
        z3Slider.value = (Math.random() * 2 - 1).toFixed(2);
        z1Val.textContent = parseFloat(z1Slider.value).toFixed(2);
        z2Val.textContent = parseFloat(z2Slider.value).toFixed(2);
        z3Val.textContent = parseFloat(z3Slider.value).toFixed(2);
        render();
    }

    function interpolate() {
        interpContainer.innerHTML = '';

        // Two random latent vectors
        const zA = [(Math.random() * 2 - 1), (Math.random() * 2 - 1), (Math.random() * 2 - 1)];
        const zB = [(Math.random() * 2 - 1), (Math.random() * 2 - 1), (Math.random() * 2 - 1)];

        const steps = 8;
        for (let s = 0; s <= steps; s++) {
            const t = s / steps;
            const z = [
                zA[0] * (1 - t) + zB[0] * t,
                zA[1] * (1 - t) + zB[1] * t,
                zA[2] * (1 - t) + zB[2] * t
            ];

            const pattern = generateFromLatent(z);
            const c = document.createElement('canvas');
            c.width = 56;
            c.height = 56;
            const cCtx = c.getContext('2d');
            cCtx.fillStyle = '#111';
            cCtx.fillRect(0, 0, 56, 56);
            renderPattern(cCtx, pattern, 0, 0, 56);
            interpContainer.appendChild(c);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
