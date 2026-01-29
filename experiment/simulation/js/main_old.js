// DCGAN Training Experiment - Main JavaScript
// Updated for digit-wise sequential training with RUN button support

// Configuration  
const CONFIG = {
	TOTAL_STEPS: 12,
	DIGITS: 10,
	ANIMATION_DELAY: 500
};

// State Management
const state = {
	currentEpoch: '25',
	completedSteps: new Set()
};

// DOM Elements
const elements = {
	epochSelect: null,
	resetBtn: null,
	trainingLog: null,
	cells: [],
	stepItems: []
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
	initializeElements();
	setupEventListeners();
});

function initializeElements() {
	elements.epochSelect = document.getElementById('epochSelect');
	elements.resetBtn = document.getElementById('resetBtn');
	elements.trainingLog = document.getElementById('trainingLog');
	elements.cells = Array.from(document.querySelectorAll('.notebook-cell'));
	elements.stepItems = Array.from(document.querySelectorAll('.step-item'));
}

function setupEventListeners() {
	// Epoch selector change
	if (elements.epochSelect) {
		elements.epochSelect.addEventListener('change', handleEpochChange);
	}

	// Run buttons - use forEach with index
	elements.cells.forEach((cell, index) => {
		const runBtn = cell.querySelector('.run-btn');
		if (runBtn) {
			runBtn.addEventListener('click', () => executeCell(index + 1));
		}
	});

	// Reset button
	if (elements.resetBtn) {
		elements.resetBtn.addEventListener('click', resetExperiment);
	}
}

function handleEpochChange(event) {
	const newEpoch = event.target.value;
	state.currentEpoch = newEpoch;
	
	// Update hyperparameters display
	updateHyperparametersDisplay();
	
	// If training step has been run, update the training output
	if (state.completedSteps.has(9)) {
		displayTrainingForCurrentEpoch();
	}
}

function executeCell(cellNumber) {
	// Validate step dependencies (except for first step)
	if (cellNumber > 1 && !state.completedSteps.has(cellNumber - 1)) {
		alert(`Please complete Step ${cellNumber - 1} first!`);
		return;
	}

	const cell = elements.cells[cellNumber - 1];
	const output = cell.querySelector('.cell-output');
	const runBtn = cell.querySelector('.run-btn');

	// Show running state
	runBtn.disabled = true;
	runBtn.innerHTML = `
		<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" class="spinner">
			<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" fill="none" stroke-dasharray="50" />
		</svg>
		Running...
	`;
	updateStepState(cellNumber, 'running');

	setTimeout(() => {
		// Execute based on cell number
		switch(cellNumber) {
			case 1:
			case 2:
			case 6:
			case 8:
				// Simple text outputs - already in HTML
				break;
			case 3:
				// Hyperparameters - update display
				updateHyperparametersDisplay();
				break;
			case 4:
			case 5:
			case 7:
				// Architecture definitions - just show success
				break;
			case 9:
				// Training loop - show combined training log
				displayTrainingForCurrentEpoch();
				break;
			case 10:
			case 11:
			case 12:
				// Charts and final results - show placeholder messages
				break;
		}

		// Show output
		if (output) {
			output.classList.remove('hidden');
		}

		// Update button to completed state
		runBtn.disabled = false;
		runBtn.innerHTML = `
			<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
				<path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/>
			</svg>
			Completed
		`;
		runBtn.classList.add('completed');

		// Mark as completed
		state.completedSteps.add(cellNumber);
		updateStepState(cellNumber, 'completed');

	}, CONFIG.ANIMATION_DELAY);
}

function displayTrainingForCurrentEpoch() {
	const data = EXPERIMENT_DATA[state.currentEpoch];
	if (!data) {
		console.error('No data found for epoch:', state.currentEpoch);
		return;
	}

	// Build combined training log for all digits
	let combinedLog = '';
	for (let digit = 0; digit < CONFIG.DIGITS; digit++) {
		const digitData = data.digitResults[digit];
		if (digitData && digitData.trainingLog) {
			combinedLog += digitData.trainingLog + '\n\n';
		}
	}

	// Add final completion message
	combinedLog += '\nALL DIGITS TRAINING COMPLETE!';

	// Display the combined log
	if (elements.trainingLog) {
		elements.trainingLog.textContent = combinedLog;
	}
}

function updateHyperparametersDisplay() {
	const codeElement = document.getElementById('hyperparametersCode');
	if (codeElement) {
		codeElement.innerHTML = `<span class="comment"># Hyperparameters</span>
latent_dim = <span class="number">100</span>
epochs_per_digit = <span class="number">${state.currentEpoch}</span>
lr = <span class="number">0.0002</span>
beta1 = <span class="number">0.5</span>`;
	}
}

function updateStepState(stepNumber, status) {
	const stepItem = elements.stepItems[stepNumber - 1];
	if (!stepItem) return;

	// Remove all status classes
	stepItem.classList.remove('active', 'running', 'completed');

	// Add new status
	switch(status) {
		case 'active':
			stepItem.classList.add('active');
			break;
		case 'running':
			stepItem.classList.add('running', 'active');
			break;
		case 'completed':
			stepItem.classList.add('completed');
			// Activate next step if exists
			if (stepNumber < CONFIG.TOTAL_STEPS) {
				const nextStep = elements.stepItems[stepNumber];
				if (nextStep && !nextStep.classList.contains('completed')) {
					nextStep.classList.add('active');
				}
			}
			break;
	}
}

function resetExperiment() {
	if (!confirm('Are you sure you want to reset the entire experiment?')) {
		return;
	}

	// Reset state
	state.completedSteps.clear();
	state.currentEpoch = '25';

	// Reset epoch selector
	if (elements.epochSelect) {
		elements.epochSelect.value = '25';
	}

	// Reset all cells
	elements.cells.forEach((cell, index) => {
		const output = cell.querySelector('.cell-output');
		if (output) {
			output.classList.add('hidden');
		}

		const runBtn = cell.querySelector('.run-btn');
		if (runBtn) {
			runBtn.disabled = false;
			runBtn.classList.remove('completed');
			runBtn.innerHTML = `
				<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
					<path d="M8 5v14l11-7z" />
				</svg>
				Run
			`;
		}

		// Reset step item states
		const stepItem = elements.stepItems[index];
		if (stepItem) {
			stepItem.classList.remove('active', 'running', 'completed');
		}
	});

	// Activate first step
	if (elements.stepItems[0]) {
		elements.stepItems[0].classList.add('active');
	}

	// Scroll to top
	window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Add CSS for spinner animation
const style = document.createElement('style');
style.textContent = `
	.spinner {
		animation: spin 1s linear infinite;
	}
	@keyframes spin {
		from { transform: rotate(0deg); }
		to { transform: rotate(360deg); }
	}
`;
document.head.appendChild(style);
