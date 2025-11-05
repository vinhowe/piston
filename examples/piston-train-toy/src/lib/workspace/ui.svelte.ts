import { config } from './config.svelte';
import { LocalStorage } from './localStorage.svelte';
import { newRun, restoreRun, type RunData } from './runs.svelte';
import {
	trainingState,
	waitForNextCheckpoint,
	workerPauseTraining,
	workerReady,
	workerRequestSave,
	workerResumeTraining,
	workerStartTraining,
	workerStep,
	workerStopTraining
} from './workers.svelte';

export const isMobile = $state({ current: false });
export const activeTab: { current: 'about' | 'metrics' } = $state({
	current: 'about'
});
export const hasWebGPU = $state({ current: false });
export const browserInfo: {
	current: {
		type:
			| 'chrome'
			| 'edge'
			| 'brave'
			| 'arc'
			| 'opera'
			| 'vivaldi'
			| 'safari'
			| 'firefox'
			| 'unknown';
		platform: 'ios' | 'macos' | 'windows' | 'android' | 'linux' | 'other';
	};
} = $state({
	current: { type: 'unknown', platform: 'other' }
});

export const setupUI = () => {

	// Browser/platform detection (best-effort; UA-CH not universally available yet)
	const ua = navigator.userAgent.toLowerCase();
	const vendor = navigator.vendor?.toLowerCase?.() ?? '';

	// Platform
	let platform: 'ios' | 'macos' | 'windows' | 'android' | 'linux' | 'other' = 'other';
	if (/iphone|ipad|ipod/.test(ua)) platform = 'ios';
	else if (/macintosh|mac os x/.test(ua)) platform = 'macos';
	else if (/windows nt/.test(ua)) platform = 'windows';
	else if (/android/.test(ua)) platform = 'android';
	else if (/linux/.test(ua)) platform = 'linux';

	// Chromium-family checks
	// Distinguish some popular Chromium variants before generic Chrome
	let type:
		| 'chrome'
		| 'edge'
		| 'brave'
		| 'arc'
		| 'opera'
		| 'vivaldi'
		| 'safari'
		| 'firefox'
		| 'unknown' = 'unknown';
	if (/edg\//.test(ua)) type = 'edge';
	else if (/vivaldi/.test(ua)) type = 'vivaldi';
	else if (/opr\//.test(ua)) type = 'opera';
	else if (/brave/.test(ua)) type = 'brave';
	else if (/arc\//.test(ua)) type = 'arc';
	else if (/firefox/.test(ua)) type = 'firefox';
	else if (/safari/.test(ua) && /apple/.test(vendor) && !/chrome|crios|android/.test(ua))
		type = 'safari';
	else if (/chrome|crios/.test(ua)) type = 'chrome';

	browserInfo.current = { type, platform };

	const mediaQuery = window.matchMedia('(min-width: 40rem)');
	isMobile.current = !mediaQuery.matches;

	// Set configOpen based on media query if not already set by user
	if (configOpen.current === null) {
		configOpen.current = mediaQuery.matches;
	}

	// Listen for changes in screen size
	const handleMediaChange = (e: MediaQueryListEvent) => {
		isMobile.current = !e.matches;

		// If switching to mobile and config is open, close it and reset tab
		if (isMobile.current && configOpen.current) {
			configOpen.current = false;
			activeTab.current = 'about';
		}
	};

	mediaQuery.addEventListener('change', handleMediaChange);

	return () => {
		mediaQuery.removeEventListener('change', handleMediaChange);
	};
};

// Function to handle tab selection with mobile behavior
export function selectTab(tabName: 'about' | 'metrics') {
	activeTab.current = tabName;
	if (isMobile.current && configOpen.current) {
		configOpen.current = false;
	}
}

let flashVramLimit = $state(false);

export function triggerVramLimitFlash() {
	controlSectionsOpen.current.training = true;

	// Scroll to GPU memory limit after a brief delay to allow section to open
	setTimeout(() => {
		const trainingVramLimitElement = document.getElementById('training-vram-limit');
		flashVramLimit = true;
		if (trainingVramLimitElement) {
			trainingVramLimitElement.scrollIntoView({
				behavior: 'instant',
				block: 'center'
			});
			trainingVramLimitElement.classList.add('error-flash');
			setTimeout(() => {
				trainingVramLimitElement.classList.remove('error-flash');
				flashVramLimit = false;
			}, 1000);
		}
	}, 100);
}

export function getFlashVramLimit() {
	return flashVramLimit;
}

let showLowDiversityDatasetError = $state(false);

export function triggerLowDiversityDatasetError() {
	controlSectionsOpen.current.task = true;
	showLowDiversityDatasetError = true;

	// Scroll to GPU memory limit after a brief delay to allow section to open
	setTimeout(() => {
		const lowDiversityDatasetErrorElement = document.getElementById('low-diversity-dataset-error');
		if (lowDiversityDatasetErrorElement) {
			lowDiversityDatasetErrorElement.scrollIntoView({
				behavior: 'instant',
				block: 'center'
			});
		}
	}, 100);
}

export function getShowLowDiversityDatasetError() {
	return showLowDiversityDatasetError;
}

export function resetLowDiversityDatasetError() {
	showLowDiversityDatasetError = false;
}

const iconStrokeWidth = $derived(isMobile ? 2 : 2.5);

export function getIconStrokeWidth() {
	return iconStrokeWidth;
}

// Initialize sectionsOpen from localStorage or use defaults
export const controlSectionsOpen = new LocalStorage('controlSectionsOpen', {
	runs: true,
	training: true,
	task: true,
	model: true,
	optimizer: true,
	advanced: false
});

export function toggleControlSection(sectionName: keyof typeof controlSectionsOpen.current) {
	controlSectionsOpen.current[sectionName] = !controlSectionsOpen.current[sectionName];
}

export const metricsSectionsOpen = new LocalStorage('metricsSectionsOpen', {});

export function toggleMetricsSection(sectionName: string) {
	metricsSectionsOpen.current[sectionName] = !(metricsSectionsOpen.current[sectionName] ?? true);
}

export const maxCompletions = new LocalStorage('maxCompletions', 4);

export function setMaxCompletions(value: number) {
	maxCompletions.current = value;
}

// Visibility state for per-metric charts (user overrides only)
export const metricVisibility = new LocalStorage('metricVisibility', {});

// Initialize configOpen from localStorage with no default (null means use media query)
export const configOpen = new LocalStorage<boolean | null>('configOpen', null);

export const tourState = new LocalStorage<{
	startedExperiment: boolean;
	restartedExperiment: boolean;
	seenCQLTutorial: boolean;
}>('tourState', {
	startedExperiment: false,
	restartedExperiment: false,
	seenCQLTutorial: false
});

export function openConfigAndScrollToControl(
	controlId: string,
	sectionsToOpen: Array<keyof typeof controlSectionsOpen.current>,
	{ useLabelFor = false }: { useLabelFor?: boolean } = {}
) {
	// Ensure left panel is open
	configOpen.current = true;
	// Open requested sections
	for (const s of sectionsToOpen) {
		controlSectionsOpen.current[s] = true;
	}
	// Scroll after a brief delay to allow layout to settle
	setTimeout(() => {
		const el = useLabelFor
			? document.querySelector(`label[for="${controlId}"]`)
			: document.getElementById(controlId);
		if (el) {
			el.scrollIntoView({ behavior: 'instant', block: 'center' });
			// Manage manual focus class
			const prev = document.querySelector('.is-focused');
			if (prev && prev !== el) prev.classList.remove('is-focused');
			el.classList.add('is-focused');
		}
	}, 100);
}

export function switchToMetrics() {
	selectTab('metrics');
}

export function toggleConfig() {
	configOpen.current = !configOpen.current;
}

export async function saveModel() {
	// Set up waiter BEFORE causing a save so auto-save on pause satisfies it
	const waiter = waitForNextCheckpoint();

	if (trainingState.current === 'training') {
		workerPauseTraining();
		// paused handler will request a save
	} else if (trainingState.current === 'paused') {
		workerRequestSave();
	} else {
		return;
	}

	const { runId, buffer } = await waiter;
	const blob = new Blob([buffer.buffer as ArrayBuffer], {
		type: 'application/octet-stream'
	});
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = `${runId}.safetensors`;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
}

// Function to start training
export function startTraining(
	options: { run?: RunData; resumeFrom: Uint8Array<ArrayBufferLike> } | undefined = undefined
) {
	const { run, resumeFrom } = options ?? {};

	if (trainingState.current !== 'stopped' || !workerReady.current) return;

	trainingState.current = 'training';
	const effectiveRun = run ? restoreRun(run) : newRun(JSON.parse(JSON.stringify(config)));

	if (isMobile.current && configOpen.current) {
		configOpen.current = false;
	}

	// We don't want to wrench them away from the visualize tab, but if they're
	// running an experiment, we want it to look like something is happening.
	if (!tourState.current.startedExperiment || activeTab.current === 'about') {
		switchToMetrics();
	}

	if (!tourState.current.startedExperiment) {
		tourState.current.startedExperiment = true;
	}

	if (getShowLowDiversityDatasetError()) {
		resetLowDiversityDatasetError();
	}

	workerStartTraining(effectiveRun.runId, resumeFrom ? resumeFrom : undefined);
}

// Function to stop training
export async function stopTraining() {
	await workerStopTraining();
}

export function togglePause() {
	if (trainingState.current === 'stopped') return;
	if (trainingState.current === 'training') {
		workerPauseTraining();
	} else {
		workerResumeTraining();
	}
}

export function stepForward() {
	if (trainingState.current === 'stopped') return;
	workerStep();
}

export async function restartTraining() {
	await stopTraining();
	startTraining();
	if (!tourState.current.restartedExperiment) {
		tourState.current.restartedExperiment = true;
	}
}
