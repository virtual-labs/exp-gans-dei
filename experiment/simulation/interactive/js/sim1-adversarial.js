// =====================================================
// Simulation 1 — Generator vs Discriminator Game (2D)
// =====================================================

(function () {
    'use strict';

    // --- Configuration ---
    const NUM_REAL_POINTS = 200;
    const NUM_GEN_POINTS = 200;
    const HEATMAP_RES = 40;

    // Canvas dimensions (set dynamically)
    let canvasW = 500, canvasH = 500, dpr = 1;

    // --- Simple neural network (2-layer) ---
    class TinyNet {
        constructor(inputDim, hiddenDim, outputDim) {
            this.w1 = this.randomMatrix(hiddenDim, inputDim);
            this.b1 = new Float32Array(hiddenDim);
            this.w2 = this.randomMatrix(outputDim, hiddenDim);
            this.b2 = new Float32Array(outputDim);
        }

        randomMatrix(rows, cols) {
            const m = [];
            for (let r = 0; r < rows; r++) {
                m[r] = new Float32Array(cols);
                for (let c = 0; c < cols; c++) {
                    m[r][c] = (Math.random() - 0.5) * 0.5;
                }
            }
            return m;
        }

        forward(x) {
            const h = new Float32Array(this.w1.length);
            for (let i = 0; i < this.w1.length; i++) {
                let sum = this.b1[i];
                for (let j = 0; j < x.length; j++) {
                    sum += this.w1[i][j] * x[j];
                }
                h[i] = Math.max(0, sum);
            }
            this._lastHidden = h;
            this._lastInput = x;

            const o = new Float32Array(this.w2.length);
            for (let i = 0; i < this.w2.length; i++) {
                let sum = this.b2[i];
                for (let j = 0; j < h.length; j++) {
                    sum += this.w2[i][j] * h[j];
                }
                o[i] = sum;
            }
            return o;
        }

        sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }

        predict(x) {
            const o = this.forward(x);
            return this.sigmoid(o[0]);
        }

        updateDiscriminator(x, label, lr) {
            const pred = this.predict(x);
            const error = pred - label;
            const dSigmoid = pred * (1 - pred);
            const dOut = error * dSigmoid;
            const h = this._lastHidden;

            for (let j = 0; j < h.length; j++) {
                this.w2[0][j] -= lr * dOut * h[j];
            }
            this.b2[0] -= lr * dOut;

            for (let j = 0; j < h.length; j++) {
                if (h[j] <= 0) continue;
                const dh = dOut * this.w2[0][j];
                for (let k = 0; k < x.length; k++) {
                    this.w1[j][k] -= lr * dh * x[k];
                }
                this.b1[j] -= lr * dh;
            }
        }
    }

    class Generator {
        constructor(noiseDim, hiddenDim) {
            this.noiseDim = noiseDim;
            this.w1 = [];
            this.b1 = new Float32Array(hiddenDim);
            this.w2 = [];
            this.b2 = new Float32Array(2);

            for (let i = 0; i < hiddenDim; i++) {
                this.w1[i] = new Float32Array(noiseDim);
                for (let j = 0; j < noiseDim; j++) {
                    this.w1[i][j] = (Math.random() - 0.5) * 0.5;
                }
            }
            for (let i = 0; i < 2; i++) {
                this.w2[i] = new Float32Array(hiddenDim);
                for (let j = 0; j < hiddenDim; j++) {
                    this.w2[i][j] = (Math.random() - 0.5) * 0.5;
                }
            }
        }

        generate(noise) {
            const h = new Float32Array(this.w1.length);
            for (let i = 0; i < this.w1.length; i++) {
                let sum = this.b1[i];
                for (let j = 0; j < noise.length; j++) {
                    sum += this.w1[i][j] * noise[j];
                }
                h[i] = Math.max(0, sum);
            }
            this._lastHidden = h;
            this._lastNoise = noise;

            const out = new Float32Array(2);
            for (let i = 0; i < 2; i++) {
                let sum = this.b2[i];
                for (let j = 0; j < h.length; j++) {
                    sum += this.w2[i][j] * h[j];
                }
                out[i] = Math.tanh(sum) * 2;
            }
            return out;
        }

        updateFromDiscriminator(disc, noise, lr) {
            const fakePoint = this.generate(noise);
            const pred = disc.predict(fakePoint);
            const dSigmoid = pred * (1 - pred);
            const dOut = -(1 - pred) * dSigmoid;

            const h = disc._lastHidden;
            const dInput = new Float32Array(2);
            for (let j = 0; j < h.length; j++) {
                if (h[j] <= 0) continue;
                const dh = dOut * disc.w2[0][j];
                for (let k = 0; k < 2; k++) {
                    dInput[k] += dh * disc.w1[j][k];
                }
            }

            const gH = this._lastHidden;
            const tanhOut = [Math.tanh(fakePoint[0] / 2), Math.tanh(fakePoint[1] / 2)];

            for (let i = 0; i < 2; i++) {
                const dtanh = (1 - tanhOut[i] * tanhOut[i]) * 2;
                const dG = dInput[i] * dtanh;
                for (let j = 0; j < gH.length; j++) {
                    this.w2[i][j] -= lr * dG * gH[j];
                }
                this.b2[i] -= lr * dG;

                for (let j = 0; j < gH.length; j++) {
                    if (gH[j] <= 0) continue;
                    const dhG = dG * this.w2[i][j];
                    for (let k = 0; k < noise.length; k++) {
                        this.w1[j][k] -= lr * dhG * noise[k];
                    }
                    this.b1[j] -= lr * dhG;
                }
            }
        }
    }

    // --- Dataset generators ---
    function generateGaussian(n) {
        const pts = [];
        for (let i = 0; i < n; i++) {
            const u1 = Math.random(), u2 = Math.random();
            const r = Math.sqrt(-2 * Math.log(u1));
            pts.push([r * Math.cos(2 * Math.PI * u2) * 0.5, r * Math.sin(2 * Math.PI * u2) * 0.5]);
        }
        return pts;
    }

    function generateCircle(n) {
        const pts = [];
        for (let i = 0; i < n; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const r = 1 + (Math.random() - 0.5) * 0.2;
            pts.push([r * Math.cos(angle), r * Math.sin(angle)]);
        }
        return pts;
    }

    function generateTwoMoons(n) {
        const pts = [];
        const half = Math.floor(n / 2);
        for (let i = 0; i < half; i++) {
            const angle = Math.PI * Math.random();
            pts.push([Math.cos(angle) + (Math.random() - 0.5) * 0.15,
                       Math.sin(angle) + (Math.random() - 0.5) * 0.15]);
        }
        for (let i = 0; i < n - half; i++) {
            const angle = Math.PI * Math.random();
            pts.push([1 - Math.cos(angle) + (Math.random() - 0.5) * 0.15,
                       0.5 - Math.sin(angle) + (Math.random() - 0.5) * 0.15]);
        }
        return pts;
    }

    // --- State ---
    let realData = [];
    let genData = [];
    let discriminator = null;
    let generator = null;
    let running = false;
    let stepCount = 0;
    let animId = null;
    let gLosses = [];
    let dLosses = [];
    let lossChart = null;

    // --- DOM refs ---
    let canvas, ctx;
    let lrGSlider, lrDSlider, gStepsSlider, dStepsSlider;
    let lrGVal, lrDVal, gStepsVal, dStepsVal;
    let datasetSelect, btnStart, btnStep, btnReset, statusDot, statusText, stepLabel;

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
        canvas = document.getElementById('sim1Canvas');
        ctx = canvas.getContext('2d');

        lrGSlider = document.getElementById('sim1LrG');
        lrDSlider = document.getElementById('sim1LrD');
        gStepsSlider = document.getElementById('sim1GSteps');
        dStepsSlider = document.getElementById('sim1DSteps');
        lrGVal = document.getElementById('sim1LrGVal');
        lrDVal = document.getElementById('sim1LrDVal');
        gStepsVal = document.getElementById('sim1GStepsVal');
        dStepsVal = document.getElementById('sim1DStepsVal');
        datasetSelect = document.getElementById('sim1Dataset');
        btnStart = document.getElementById('sim1Start');
        btnStep = document.getElementById('sim1Step');
        btnReset = document.getElementById('sim1Reset');
        statusDot = document.getElementById('sim1StatusDot');
        statusText = document.getElementById('sim1StatusText');
        stepLabel = document.getElementById('sim1StepCount');

        lrGSlider.oninput = () => lrGVal.textContent = parseFloat(lrGSlider.value).toFixed(4);
        lrDSlider.oninput = () => lrDVal.textContent = parseFloat(lrDSlider.value).toFixed(4);
        gStepsSlider.oninput = () => gStepsVal.textContent = gStepsSlider.value;
        dStepsSlider.oninput = () => dStepsVal.textContent = dStepsSlider.value;

        btnStart.addEventListener('click', toggleRun);
        btnStep.addEventListener('click', stepOnce);
        btnReset.addEventListener('click', reset);
        datasetSelect.addEventListener('change', reset);

        // Init loss chart
        const chartCanvas = document.getElementById('sim1LossChart');
        lossChart = new Chart(chartCanvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'G Loss', data: [], borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.3 },
                    { label: 'D Loss', data: [], borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: true, title: { display: true, text: 'Step', font: { size: 10 } }, ticks: { maxTicksLimit: 6, font: { size: 9 } } },
                    y: { display: true, title: { display: true, text: 'Loss', font: { size: 10 } }, ticks: { font: { size: 9 } } }
                },
                plugins: {
                    legend: { labels: { font: { size: 10 } } }
                }
            }
        });

        // Size canvas after layout settles
        requestAnimationFrame(() => { sizeCanvas(); reset(); });
        window.addEventListener('resize', () => { sizeCanvas(); draw(); });
    }

    function reset() {
        running = false;
        if (animId) cancelAnimationFrame(animId);
        animId = null;
        stepCount = 0;
        gLosses = [];
        dLosses = [];

        btnStart.textContent = '▶ Start';
        statusDot.className = 'status-dot';
        statusText.textContent = 'Ready';
        stepLabel.textContent = 'Step: 0';

        discriminator = new TinyNet(2, 32, 1);
        generator = new Generator(8, 32);

        const ds = datasetSelect.value;
        if (ds === 'gaussian') realData = generateGaussian(NUM_REAL_POINTS);
        else if (ds === 'circle') realData = generateCircle(NUM_REAL_POINTS);
        else realData = generateTwoMoons(NUM_REAL_POINTS);

        updateGenData();

        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.data.datasets[1].data = [];
        lossChart.update();

        draw();
    }

    function updateGenData() {
        genData = [];
        for (let i = 0; i < NUM_GEN_POINTS; i++) {
            const noise = new Float32Array(generator.noiseDim);
            for (let j = 0; j < noise.length; j++) noise[j] = (Math.random() - 0.5) * 2;
            genData.push(generator.generate(noise));
        }
    }

    function randomNoise(dim) {
        const n = new Float32Array(dim);
        for (let i = 0; i < dim; i++) n[i] = (Math.random() - 0.5) * 2;
        return n;
    }

    function trainStep() {
        const lrG = parseFloat(lrGSlider.value);
        const lrD = parseFloat(lrDSlider.value);
        const gSteps = parseInt(gStepsSlider.value);
        const dSteps = parseInt(dStepsSlider.value);

        let dLoss = 0;
        for (let s = 0; s < dSteps; s++) {
            for (let i = 0; i < 16; i++) {
                const idx = Math.floor(Math.random() * realData.length);
                discriminator.updateDiscriminator(realData[idx], 1, lrD);
                const p = discriminator.predict(realData[idx]);
                dLoss += -Math.log(Math.max(1e-7, p));
            }
            for (let i = 0; i < 16; i++) {
                const noise = randomNoise(generator.noiseDim);
                const fakePoint = generator.generate(noise);
                discriminator.updateDiscriminator(fakePoint, 0, lrD);
                const p = discriminator.predict(fakePoint);
                dLoss += -Math.log(Math.max(1e-7, 1 - p));
            }
        }
        dLoss /= (dSteps * 32);

        let gLoss = 0;
        for (let s = 0; s < gSteps; s++) {
            for (let i = 0; i < 16; i++) {
                const noise = randomNoise(generator.noiseDim);
                generator.updateFromDiscriminator(discriminator, noise, lrG);
                const fakePoint = generator.generate(noise);
                const p = discriminator.predict(fakePoint);
                gLoss += -Math.log(Math.max(1e-7, p));
            }
        }
        gLoss /= (gSteps * 16);

        stepCount++;
        gLosses.push(gLoss);
        dLosses.push(dLoss);

        if (stepCount % 2 === 0) {
            lossChart.data.labels.push(stepCount);
            lossChart.data.datasets[0].data.push(gLoss);
            lossChart.data.datasets[1].data.push(dLoss);
            if (lossChart.data.labels.length > 100) {
                lossChart.data.labels.shift();
                lossChart.data.datasets[0].data.shift();
                lossChart.data.datasets[1].data.shift();
            }
            lossChart.update();
        }

        stepLabel.textContent = `Step: ${stepCount}`;
        updateGenData();
    }

    function stepOnce() {
        trainStep();
        draw();
    }

    function toggleRun() {
        running = !running;
        if (running) {
            btnStart.textContent = '⏸ Pause';
            statusDot.className = 'status-dot running';
            statusText.textContent = 'Training';
            runLoop();
        } else {
            btnStart.textContent = '▶ Start';
            statusDot.className = 'status-dot paused';
            statusText.textContent = 'Paused';
            if (animId) cancelAnimationFrame(animId);
        }
    }

    function runLoop() {
        if (!running) return;
        trainStep();
        draw();
        animId = requestAnimationFrame(runLoop);
    }

    // --- Drawing (uses CSS pixel dimensions) ---
    function toCanvasX(x) { return (x + 3) / 6 * canvasW; }
    function toCanvasY(y) { return (3 - y) / 6 * canvasH; }

    function draw() {
        ctx.clearRect(0, 0, canvasW, canvasH);

        // Heatmap
        const cellW = canvasW / HEATMAP_RES;
        const cellH = canvasH / HEATMAP_RES;
        for (let i = 0; i < HEATMAP_RES; i++) {
            for (let j = 0; j < HEATMAP_RES; j++) {
                const x = (i / HEATMAP_RES) * 6 - 3;
                const y = 3 - (j / HEATMAP_RES) * 6;
                const conf = discriminator.predict([x, y]);
                const r = Math.floor((1 - conf) * 200 + 40);
                const g = Math.floor(conf * 80 + 40);
                const b = Math.floor(conf * 200 + 40);
                ctx.fillStyle = `rgba(${r},${g},${b},0.25)`;
                ctx.fillRect(i * cellW, j * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        // Dot radius scales with canvas
        const dotR = Math.max(2.5, Math.min(5, canvasW / 180));

        // Real data (blue)
        ctx.fillStyle = '#3b82f6';
        for (const p of realData) {
            ctx.beginPath();
            ctx.arc(toCanvasX(p[0]), toCanvasY(p[1]), dotR, 0, Math.PI * 2);
            ctx.fill();
        }

        // Generated data (red)
        ctx.fillStyle = '#ef4444';
        for (const p of genData) {
            ctx.beginPath();
            ctx.arc(toCanvasX(p[0]), toCanvasY(p[1]), dotR, 0, Math.PI * 2);
            ctx.fill();
        }

        // Legend
        const fs = Math.max(10, canvasW / 60);
        ctx.font = `${fs}px Inter, sans-serif`;
        ctx.fillStyle = '#3b82f6';
        ctx.fillRect(10, 10, 10, 10);
        ctx.fillStyle = '#333';
        ctx.fillText('Real Data', 25, 20);

        ctx.fillStyle = '#ef4444';
        ctx.fillRect(10, 28, 10, 10);
        ctx.fillStyle = '#333';
        ctx.fillText('Generated Data', 25, 38);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
