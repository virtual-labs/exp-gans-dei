// =====================================================
// Simulation 3 — Mode Collapse Playground
// =====================================================

(function () {
    'use strict';

    const NUM_CLUSTERS = 6;
    const POINTS_PER_CLUSTER = 30;

    // Canvas dimensions (set dynamically)
    let canvasW = 500, canvasH = 500, dpr = 1;


    // --- Simple Generator & Discriminator ---
    class SimpleNet {
        constructor(inDim, hidden, outDim) {
            this.w1 = this.rand(hidden, inDim);
            this.b1 = new Float32Array(hidden);
            this.w2 = this.rand(outDim, hidden);
            this.b2 = new Float32Array(outDim);
        }
        rand(r, c) {
            const m = [];
            for (let i = 0; i < r; i++) {
                m[i] = new Float32Array(c);
                for (let j = 0; j < c; j++) m[i][j] = (Math.random() - 0.5) * 0.4;
            }
            return m;
        }
        forward(x, useBN) {
            const h = new Float32Array(this.w1.length);
            for (let i = 0; i < this.w1.length; i++) {
                let s = this.b1[i];
                for (let j = 0; j < x.length; j++) s += this.w1[i][j] * x[j];
                h[i] = Math.max(0, s);
            }
            if (useBN) {
                let mean = 0, std = 0;
                for (let i = 0; i < h.length; i++) mean += h[i];
                mean /= h.length;
                for (let i = 0; i < h.length; i++) std += (h[i] - mean) ** 2;
                std = Math.sqrt(std / h.length + 1e-5);
                for (let i = 0; i < h.length; i++) h[i] = (h[i] - mean) / std;
            }
            this._h = h;
            this._x = x;
            const o = new Float32Array(this.w2.length);
            for (let i = 0; i < this.w2.length; i++) {
                let s = this.b2[i];
                for (let j = 0; j < h.length; j++) s += this.w2[i][j] * h[j];
                o[i] = s;
            }
            return o;
        }
    }

    // --- State ---
    let realClusters = [];
    let realPoints = [];
    let genPoints = [];
    let gen = null;
    let disc = null;
    let running = false;
    let stepCount = 0;
    let animId = null;
    let useBN = false;
    let useWGAN = false;
    let noiseDim = 8;
    let genLR = 0.005;

    let canvas, ctx;
    let btnStart, btnReset;
    let diversityBar, modeCountEl, totalModesEl, stepLabel;

    function generateClusters() {
        realClusters = [];
        realPoints = [];
        const angleStep = (2 * Math.PI) / NUM_CLUSTERS;
        for (let c = 0; c < NUM_CLUSTERS; c++) {
            const cx = Math.cos(angleStep * c) * 1.5;
            const cy = Math.sin(angleStep * c) * 1.5;
            realClusters.push([cx, cy]);
            for (let i = 0; i < POINTS_PER_CLUSTER; i++) {
                const u1 = Math.random(), u2 = Math.random();
                const r = Math.sqrt(-2 * Math.log(u1));
                realPoints.push([
                    cx + r * Math.cos(2 * Math.PI * u2) * 0.15,
                    cy + r * Math.sin(2 * Math.PI * u2) * 0.15
                ]);
            }
        }
    }

    function sizeCanvas() {
        const container = canvas.parentElement;
        const rect = container.getBoundingClientRect();
        dpr = window.devicePixelRatio || 1;
        canvasW = rect.width;
        canvasH = rect.height;
        canvas.width = canvasW * dpr;
        canvas.height = canvasH * dpr;
        canvas.style.width = canvasW + 'px';
        canvas.style.height = canvasH + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function init() {
        canvas = document.getElementById('sim3Canvas');
        ctx = canvas.getContext('2d');

        btnStart = document.getElementById('sim3Start');
        btnReset = document.getElementById('sim3Reset');
        diversityBar = document.getElementById('sim3DiversityFill');
        modeCountEl = document.getElementById('sim3ModeCount');
        totalModesEl = document.getElementById('sim3TotalModes');
        stepLabel = document.getElementById('sim3StepCount');

        const lrSlider = document.getElementById('sim3LrG');
        const lrVal = document.getElementById('sim3LrGVal');
        const ndSlider = document.getElementById('sim3NoiseDim');
        const ndVal = document.getElementById('sim3NoiseDimVal');
        const bnToggle = document.getElementById('sim3BN');
        const wganToggle = document.getElementById('sim3WGAN');

        lrSlider.oninput = () => {
            genLR = parseFloat(lrSlider.value);
            lrVal.textContent = genLR.toFixed(4);
        };
        ndSlider.oninput = () => {
            noiseDim = parseInt(ndSlider.value);
            ndVal.textContent = noiseDim;
        };
        bnToggle.onchange = () => { useBN = bnToggle.checked; };
        wganToggle.onchange = () => { useWGAN = wganToggle.checked; };

        btnStart.addEventListener('click', toggleRun);
        btnReset.addEventListener('click', reset);

        totalModesEl.textContent = NUM_CLUSTERS;
        requestAnimationFrame(() => { sizeCanvas(); reset(); });
        window.addEventListener('resize', () => { sizeCanvas(); draw(); });
    }

    function reset() {
        running = false;
        if (animId) cancelAnimationFrame(animId);
        animId = null;
        stepCount = 0;

        btnStart.textContent = '▶ Start';
        stepLabel.textContent = 'Step: 0';

        generateClusters();

        gen = new SimpleNet(noiseDim, 32, 2);
        disc = new SimpleNet(2, 32, 1);

        updateGenData();
        updateDiversity();
        draw();
    }

    function randomNoise() {
        const n = new Float32Array(noiseDim);
        for (let i = 0; i < noiseDim; i++) n[i] = (Math.random() - 0.5) * 2;
        return n;
    }

    function updateGenData() {
        genPoints = [];
        for (let i = 0; i < 150; i++) {
            const noise = randomNoise();
            const out = gen.forward(noise, useBN);
            genPoints.push([Math.tanh(out[0]) * 2.5, Math.tanh(out[1]) * 2.5]);
        }
    }

    function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }

    function trainStep() {
        // Train discriminator
        for (let i = 0; i < 12; i++) {
            const realIdx = Math.floor(Math.random() * realPoints.length);
            const realOut = disc.forward(realPoints[realIdx], false);
            const pReal = sigmoid(realOut[0]);
            const errReal = pReal - 1;
            // Simple SGD on disc
            for (let j = 0; j < disc._h.length; j++) {
                if (disc._h[j] <= 0) continue;
                const dh = errReal * pReal * (1 - pReal) * disc.w2[0][j];
                for (let k = 0; k < 2; k++) {
                    disc.w1[j][k] -= 0.01 * dh * disc._x[k];
                }
                disc.b1[j] -= 0.01 * dh;
            }
            disc.w2[0].forEach((_, j) => {
                disc.w2[0][j] -= 0.01 * errReal * pReal * (1 - pReal) * disc._h[j];
            });

            // Fake
            const noise = randomNoise();
            const fakePoint = gen.forward(noise, useBN);
            const fp = [Math.tanh(fakePoint[0]) * 2.5, Math.tanh(fakePoint[1]) * 2.5];
            const fakeOut = disc.forward(fp, false);
            const pFake = sigmoid(fakeOut[0]);
            const errFake = pFake - 0;
            for (let j = 0; j < disc._h.length; j++) {
                if (disc._h[j] <= 0) continue;
                const dh = errFake * pFake * (1 - pFake) * disc.w2[0][j];
                for (let k = 0; k < 2; k++) {
                    disc.w1[j][k] -= 0.01 * dh * disc._x[k];
                }
                disc.b1[j] -= 0.01 * dh;
            }
            disc.w2[0].forEach((_, j) => {
                disc.w2[0][j] -= 0.01 * errFake * pFake * (1 - pFake) * disc._h[j];
            });
        }

        // Train generator (simplified gradient push)
        for (let i = 0; i < 8; i++) {
            const noise = randomNoise();
            const fakePoint = gen.forward(noise, useBN);
            const fp = [Math.tanh(fakePoint[0]) * 2.5, Math.tanh(fakePoint[1]) * 2.5];
            const fakeOut = disc.forward(fp, false);
            const pFake = sigmoid(fakeOut[0]);

            // Get gradient direction from discriminator
            const dInput = new Float32Array(2);
            for (let j = 0; j < disc._h.length; j++) {
                if (disc._h[j] <= 0) continue;
                for (let k = 0; k < 2; k++) {
                    dInput[k] += disc.w2[0][j] * disc.w1[j][k];
                }
            }

            // Push generator output in direction that increases D(G(z))
            const gH = gen._h;
            for (let d = 0; d < 2; d++) {
                const grad = dInput[d] * genLR;
                for (let j = 0; j < gH.length; j++) {
                    gen.w2[d][j] += grad * gH[j] * 0.1;
                }
                gen.b2[d] += grad * 0.1;
            }
        }

        stepCount++;
        stepLabel.textContent = `Step: ${stepCount}`;
        updateGenData();
        updateDiversity();
    }

    function updateDiversity() {
        // Count how many clusters the generator covers
        let coveredModes = 0;
        for (const [cx, cy] of realClusters) {
            let nearby = 0;
            for (const [gx, gy] of genPoints) {
                const dist = Math.sqrt((gx - cx) ** 2 + (gy - cy) ** 2);
                if (dist < 0.5) nearby++;
            }
            if (nearby >= 3) coveredModes++;
        }
        modeCountEl.textContent = coveredModes;

        const diversity = coveredModes / NUM_CLUSTERS;
        diversityBar.style.width = (diversity * 100) + '%';
    }

    function toggleRun() {
        running = !running;
        if (running) {
            btnStart.textContent = '⏸ Pause';
            runLoop();
        } else {
            btnStart.textContent = '▶ Start';
            if (animId) cancelAnimationFrame(animId);
        }
    }

    function runLoop() {
        if (!running) return;
        trainStep();
        draw();
        animId = requestAnimationFrame(runLoop);
    }

    function toCanvasX(x) { return (x + 3) / 6 * canvasW; }
    function toCanvasY(y) { return (3 - y) / 6 * canvasH; }

    function draw() {
        ctx.clearRect(0, 0, canvasW, canvasH);

        // Background grid
        ctx.strokeStyle = '#f0f0f0';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 6; i++) {
            const px = (i / 6) * canvasW;
            const py = (i / 6) * canvasH;
            ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, canvasH); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(canvasW, py); ctx.stroke();
        }

        // Cluster circles (light background)
        const clusterR = Math.max(20, Math.min(50, canvasW / 20));
        ctx.globalAlpha = 0.1;
        for (const [cx, cy] of realClusters) {
            ctx.beginPath();
            ctx.arc(toCanvasX(cx), toCanvasY(cy), clusterR, 0, Math.PI * 2);
            ctx.fillStyle = '#3b82f6';
            ctx.fill();
        }
        ctx.globalAlpha = 1;

        const dotR = Math.max(2, Math.min(4.5, canvasW / 200));

        // Real points (blue)
        ctx.fillStyle = '#3b82f6';
        for (const p of realPoints) {
            ctx.beginPath();
            ctx.arc(toCanvasX(p[0]), toCanvasY(p[1]), dotR, 0, Math.PI * 2);
            ctx.fill();
        }

        // Generated points (red)
        ctx.fillStyle = '#ef4444';
        for (const p of genPoints) {
            ctx.beginPath();
            ctx.arc(toCanvasX(p[0]), toCanvasY(p[1]), dotR * 1.2, 0, Math.PI * 2);
            ctx.fill();
        }

        // Legend
        const fs = Math.max(10, Math.min(13, canvasW / 60));
        const boxS = Math.round(fs * 0.85);
        const pad = 10;
        const lineH = Math.round(fs * 1.6);
        ctx.font = `${fs}px Inter, sans-serif`;

        ctx.fillStyle = '#3b82f6';
        ctx.fillRect(pad, pad, boxS, boxS);
        ctx.fillStyle = '#333';
        ctx.fillText('Real Clusters', pad + boxS + 6, pad + boxS - 1);

        ctx.fillStyle = '#ef4444';
        ctx.fillRect(pad, pad + lineH, boxS, boxS);
        ctx.fillStyle = '#333';
        ctx.fillText('Generator Output', pad + boxS + 6, pad + lineH + boxS - 1);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
