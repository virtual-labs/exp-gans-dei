// =====================================================
// Simulation 2 — Real vs Generated Panel (Epoch Evolution)
// =====================================================

(function () {
    'use strict';

    let gaugeCharts = {};
    let autoPlayInterval = null;

    function init() {
        const epochSlider = document.getElementById('sim2EpochSlider');
        const epochVal = document.getElementById('sim2EpochVal');
        const digitSelect = document.getElementById('sim2Digit');
        const configSelect = document.getElementById('sim2Config');
        const btnAutoPlay = document.getElementById('sim2AutoPlay');
        const realPanel = document.getElementById('sim2RealImages');
        const fakePanel = document.getElementById('sim2FakeImages');

        // Epoch checkpoints
        const epochs25 = [1, 5, 10, 15, 20, 25];
        const epochs30 = [1, 5, 10, 15, 20, 25, 30];

        function getEpochs() {
            return configSelect.value === '25' ? epochs25 : epochs30;
        }

        function getMaxEpoch() {
            return parseInt(configSelect.value);
        }

        function snapToNearestEpoch(val) {
            const epochs = getEpochs();
            let closest = epochs[0];
            let minDiff = Math.abs(val - closest);
            for (const e of epochs) {
                if (Math.abs(val - e) < minDiff) {
                    minDiff = Math.abs(val - e);
                    closest = e;
                }
            }
            return closest;
        }

        function updateDisplay() {
            const digit = digitSelect.value;
            const epoch = snapToNearestEpoch(parseInt(epochSlider.value));
            const config = configSelect.value;

            epochVal.textContent = `Epoch ${epoch}/${config}`;
            epochSlider.value = epoch;

            // Show training images (real vs fake is in combined images)
            const imgPath = `../notebook/images/training_digit${digit}_epoch${epoch}_${config}epochs.png`;

            realPanel.innerHTML = `
                <img src="../notebook/images/training_digit${digit}_epoch1_${config}epochs.png"
                     alt="Real Digit ${digit}" style="max-width:100%; image-rendering: pixelated; border-radius: 4px;">
                <p style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Real MNIST Digit ${digit}</p>
            `;

            fakePanel.innerHTML = `
                <img src="${imgPath}"
                     alt="Generated Digit ${digit} at Epoch ${epoch}"
                     onerror="this.src='../notebook/images/training_digit${digit}_epoch1_${config}epochs.png'"
                     style="max-width:100%; image-rendering: pixelated; border-radius: 4px;">
                <p style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Epoch ${epoch} / ${config}</p>
            `;

            // Update gauges
            const progress = epoch / getMaxEpoch();
            updateGauge('diversity', 0.3 + progress * 0.5);
            updateGauge('sharpness', 0.1 + progress * 0.7);
            updateGauge('dAccuracy', Math.max(0.3, 0.95 - progress * 0.45));
        }

        function updateGauge(name, value) {
            const valEl = document.getElementById(`sim2Gauge${capitalize(name)}Val`);
            if (valEl) valEl.textContent = (value * 100).toFixed(0) + '%';

            if (gaugeCharts[name]) {
                gaugeCharts[name].data.datasets[0].data = [value, 1 - value];
                gaugeCharts[name].update();
            }
        }

        function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

        // Init gauges
        ['diversity', 'sharpness', 'dAccuracy'].forEach((name, i) => {
            const canvasEl = document.getElementById(`sim2Gauge${capitalize(name)}`);
            if (!canvasEl) return;
            const colors = ['#22c55e', '#6366f1', '#f59e0b'];
            gaugeCharts[name] = new Chart(canvasEl, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [0.5, 0.5],
                        backgroundColor: [colors[i], '#e5e7eb'],
                        borderWidth: 0
                    }]
                },
                options: {
                    cutout: '70%',
                    responsive: true,
                    maintainAspectRatio: true,
                    animation: { duration: 300 },
                    plugins: { legend: { display: false }, tooltip: { enabled: false } }
                }
            });
        });

        // Events
        epochSlider.addEventListener('input', updateDisplay);
        digitSelect.addEventListener('change', updateDisplay);
        configSelect.addEventListener('change', () => {
            const max = getMaxEpoch();
            epochSlider.max = max;
            epochSlider.value = Math.min(parseInt(epochSlider.value), max);
            updateDisplay();
        });

        btnAutoPlay.addEventListener('click', () => {
            if (autoPlayInterval) {
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
                btnAutoPlay.textContent = '▶ Auto Play';
                return;
            }

            btnAutoPlay.textContent = '⏸ Stop';
            const epochs = getEpochs();
            let idx = 0;

            autoPlayInterval = setInterval(() => {
                if (idx >= epochs.length) {
                    clearInterval(autoPlayInterval);
                    autoPlayInterval = null;
                    btnAutoPlay.textContent = '▶ Auto Play';
                    return;
                }
                epochSlider.value = epochs[idx];
                updateDisplay();
                idx++;
            }, 1200);
        });

        updateDisplay();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
