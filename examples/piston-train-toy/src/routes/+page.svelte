<script lang="ts">
	import ActionButton from '$lib/components/ActionButton.svelte';
	import Controls from '$lib/components/controls/Controls.svelte';
	import { config, initSharedConfigUrlSync, replaceConfig } from '$lib/workspace/config.svelte';
	import { lastSessionStore } from '$lib/workspace/lastSessionStore';
	import { currentRun, resetWorkspace, restoreRun, runCounter } from '$lib/workspace/runs.svelte';
	import {
		activeTab,
		configOpen,
		getIconStrokeWidth,
		hasWebGPU,
		isMobile,
		restartTraining,
		saveModel,
		selectTab,
		setupUI,
		startTraining,
		stepForward,
		stopTraining,
		toggleConfig,
		togglePause,
		tourState
	} from '$lib/workspace/ui.svelte';
	import {
		cleanupWorkers,
		initializeWorkers,
		peekCheckpointConfig,
		trainingState,
		workerReady
	} from '$lib/workspace/workers.svelte';
	import {
		ChartLine,
		DownloadIcon,
		ExternalLink,
		HistoryIcon,
		Info,
		PauseIcon,
		PlayIcon,
		RefreshCcwIcon,
		Settings2,
		SquareIcon,
		StepForwardIcon
	} from 'lucide-svelte';
	import { onDestroy, onMount } from 'svelte';

	import About from './tabs/About.svelte';
	import Metrics from './tabs/Metrics.svelte';

	// Mount state to prevent flash
	let hasMounted = $state(false);

	let shouldShowTabContent = $derived(!isMobile.current || !configOpen.current);

	const iconStrokeWidth = $derived(getIconStrokeWidth());

	// Drag & drop / paste import state
	let isDragOver = $state(false);
	let dragDepth = $state(0);
	let showReplaceDialog = $state(false);
	let pendingImportBytes = $state<ArrayBuffer | null>(null);
	let hasLastSession = $state(false);
	let canResume = $derived(hasLastSession && runCounter.current === 0);

	function onBackdropClick(e: MouseEvent) {
		if (e.currentTarget === e.target) cancelReplace();
	}

	function onKeyDown(e: KeyboardEvent) {
		if (e.key === 'Escape' && showReplaceDialog) {
			showReplaceDialog = false;
			pendingImportBytes = null;
		}
	}

	async function computeHasLastSession() {
		hasLastSession = await lastSessionStore.exists();
	}

	async function handleResumeClick() {
		if (trainingState.current !== 'stopped') return;
		const lastSession = await lastSessionStore.get();
		if (!lastSession) return;

		// Replace config, restore run, select metrics tab, and start training
		replaceConfig(lastSession.run.config);
		restoreRun(lastSession.run);
		selectTab('metrics');
		startTraining({ run: lastSession.run, resumeFrom: lastSession.checkpoint });
	}

	async function handleFileImport(file: File) {
		const buf = await file.arrayBuffer();
		await maybeStartFromBytes(buf);
	}

	async function startFromBytes(bytes: ArrayBuffer) {
		const cfg = await peekCheckpointConfig(new Uint8Array(bytes));
		replaceConfig(cfg);
		startTraining({ resumeFrom: new Uint8Array(bytes) });
	}

	async function maybeStartFromBytes(bytes: ArrayBuffer) {
		if (trainingState.current !== 'stopped' && currentRun.current) {
			pendingImportBytes = bytes;
			showReplaceDialog = true;
			return;
		}
		// Peek config from checkpoint and replace UI config before starting
		await startFromBytes(bytes);
	}

	function confirmReplace() {
		const bytes = pendingImportBytes;
		pendingImportBytes = null;
		showReplaceDialog = false;
		if (bytes) {
			// We were in a training session; stop, peek config, replace, then start from checkpoint
			stopTraining().then(() => startFromBytes(bytes));
		}
	}

	function cancelReplace() {
		pendingImportBytes = null;
		showReplaceDialog = false;
	}

	// Check if current config is different from the config used to start the run
	const shouldSuggestRestart = $derived.by(() => {
		if (!currentRun.current?.config || !config) return false;
		// We can hot-update the visualization config without restarting the run, so we exclude it
		const { visualization: _1, ...newConfig } = config;
		const { visualization: _2, ...runConfig } = currentRun.current.config;
		return JSON.stringify(newConfig) !== JSON.stringify(runConfig);
	});

	onMount(() => {
		// Set mounted state to remove hidden classes
		hasMounted = true;

		initSharedConfigUrlSync();
		resetWorkspace();
		// Check for WebGPU support
		void (async () => {
			const hasGPUInNavigator = 'gpu' in navigator;
			if (!hasGPUInNavigator) {
				hasWebGPU.current = false;
				console.log('No WebGPU support found in navigator');
				return;
			}

			try {
				hasWebGPU.current = !!(await navigator.gpu.requestAdapter());
				if (hasWebGPU.current) {
					initializeWorkers();
					console.log('WebGPU support found in navigator');
				} else {
					console.log('WebGPU adapter request returned falsy value; disabling WebGPU support');
				}
			} catch (_error) {
				hasWebGPU.current = false;
				console.log('Error requesting WebGPU adapter; disabling WebGPU support: ', _error);
			}
		})();

		const uiCleanup = setupUI();

		void computeHasLastSession();

		// Global drag & drop handlers
		const onDragEnter = (e: DragEvent) => {
			e.preventDefault();
			dragDepth += 1;
		};
		const onDragOver = (e: DragEvent) => {
			e.preventDefault();
			isDragOver = true;
		};
		const onDragLeave = (e: DragEvent) => {
			e.preventDefault();
			dragDepth = Math.max(0, dragDepth - 1);
			if (dragDepth === 0) isDragOver = false;
		};
		const onDrop = (e: DragEvent) => {
			e.preventDefault();
			isDragOver = false;
			dragDepth = 0;
			const files = e.dataTransfer?.files;
			if (!files || files.length === 0) return;
			const file = files[0]!;
			void handleFileImport(file);
		};

		// Paste handler for clipboard file
		const onPaste = (e: ClipboardEvent) => {
			const items = e.clipboardData?.items;
			if (!items) return;
			for (const it of items) {
				const f = it.getAsFile?.();
				if (f) {
					void handleFileImport(f);
					break;
				}
			}
		};

		window.addEventListener('dragenter', onDragEnter);
		window.addEventListener('dragover', onDragOver);
		window.addEventListener('dragleave', onDragLeave);
		window.addEventListener('drop', onDrop);
		window.addEventListener('paste', onPaste);
		window.addEventListener('keydown', onKeyDown);

		return () => {
			uiCleanup();
			window.removeEventListener('dragenter', onDragEnter);
			window.removeEventListener('dragover', onDragOver);
			window.removeEventListener('dragleave', onDragLeave);
			window.removeEventListener('drop', onDrop);
			window.removeEventListener('paste', onPaste);
			window.removeEventListener('keydown', onKeyDown);
		};
	});

	onDestroy(() => {
		cleanupWorkers();
	});
</script>

{#snippet tabButton(
	title: string,
	icon: typeof ChartLine,
	hideBorder: boolean,
	isActive: boolean,
	disabled: boolean,
	highlighted: boolean,
	hasActivity: boolean,
	onClick: () => void
)}
	{@const Icon = icon}
	<button
		type="button"
		class="px-2 py-1 flex items-center gap-1.5 text-base border-b border-r border-panel-border-base {isActive
			? 'bg-white' + (hideBorder ? ' border-b-transparent' : '')
			: highlighted
				? 'bg-gradient-to-t from-purple-800 to-purple-400 text-purple-100 animate-pulse font-medium'
				: 'bg-panel'} {disabled ? 'opacity-50' : 'cursor-pointer'}"
		onclick={onClick}
		{disabled}
	>
		{#if hasActivity}
			<span class="w-3.5 h-3.5 flex items-center justify-center">
				<span
					class="w-2 h-2 border border-green-800 bg-green-500 rounded-full animate-stepped-pulse translate-y-[0.5px]"
				></span>
			</span>
		{:else}
			<Icon class="w-3.5 h-3.5 translate-y-[0.25px]" strokeWidth={iconStrokeWidth} />
		{/if}
		{title}
	</button>
{/snippet}

<main class="h-full w-full flex flex-col relative">
	<div
		class="w-full py-0.5 px-1.5 flex items-center justify-between text-purple-900 bg-purple-200 border-b border-purple-300 gap-8"
	>
		<span class="uppercase font-mono font-semibold tracking-wider text-xs"> Sequence Toy </span>
		<div class="flex items-center gap-3 font-medium">
			<a
				href="https://github.com/vinhowe/piston"
				target="_blank"
				rel="noopener noreferrer"
				class="flex items-center gap-1 underline"
			>
				Github <ExternalLink class="inline-block h-3.5 w-3.5" strokeWidth={iconStrokeWidth} />
			</a>
		</div>
	</div>
	<div
		class="w-full flex-1 min-h-0 relative flex flex-col sm:grid sm:grid-cols-[max-content_1fr] sm:grid-rows-[min-content_1fr] overflow-x-hidden"
	>
		<div
			class="flex sm:grid sm:grid-cols-subgrid sm:col-span-2 min-w-0 overflow-x-auto bg-panel"
			aria-label="Tab navigation"
			role="navigation"
			data-nosnippet
		>
			<div class="bg-panel border-b border-r border-panel-border-base flex h-full">
				<div class="-mb-px flex">
					{@render tabButton(
						'Experiment',
						Settings2,
						true,
						configOpen.current !== false,
						!hasWebGPU.current,
						hasWebGPU.current &&
							((runCounter.current === 0 && !tourState.current.startedExperiment) ||
								(shouldSuggestRestart && !tourState.current.restartedExperiment)),
						false,
						toggleConfig
					)}
				</div>
				<div class="flex-1 bg-panel w-1.5"></div>
			</div>

			<!-- Tab Headers -->
			<div class="bg-panel h-min flex flex-1">
				{@render tabButton(
					'About',
					Info,
					true,
					activeTab.current === 'about' && shouldShowTabContent,
					false,
					false,
					false,
					() => selectTab('about')
				)}
				{@render tabButton(
					'Metrics',
					ChartLine,
					true,
					activeTab.current === 'metrics' && shouldShowTabContent,
					runCounter.current === 0,
					false,
					trainingState.current === 'training',
					() => selectTab('metrics')
				)}
				<div class="flex-1 border-b border-panel-border-base"></div>
			</div>
		</div>

		<!-- Left Panel: Controls -->
		{#if configOpen.current !== false}
			<div
				class="sm:w-75 sm:border-r border-panel-border-base min-h-0 flex-1 shrink-0 relative overflow-hidden {hasMounted
					? ''
					: 'hidden sm:block'}"
			>
				<div class="relative flex flex-col h-full">
					{#if trainingState.current !== 'stopped'}
						<div
							class="p-1 border-b border-panel-border-base shrink-0 flex items-center justify-between"
						>
							<span class="font-medium">{currentRun.current?.runId}</span>
							<button class="flex items-center gap-1 cursor-pointer" onclick={saveModel}>
								<DownloadIcon class="w-3.5 h-3.5 shrink-0" strokeWidth={iconStrokeWidth} />
							</button>
						</div>
					{/if}
					<div class="p-1 border-b border-panel-border-base shrink-0">
						<div class="space-y-1">
							{#if trainingState.current === 'stopped'}
								{#if canResume}
									<ActionButton
										color="purple"
										class="w-full h-7.5"
										disabled={trainingState.current !== 'stopped' || !workerReady.current}
										onclick={handleResumeClick}
									>
										<span class="flex items-center justify-center gap-1.5 w-full">
											<HistoryIcon class="w-3.5 h-3.5 shrink-0" strokeWidth={iconStrokeWidth} />
											Resume from last session
										</span>
									</ActionButton>
								{/if}
								<ActionButton
									color="green"
									disabled={trainingState.current !== 'stopped' || !workerReady.current}
									onclick={() => startTraining()}
									highlighted={workerReady.current &&
										trainingState.current === 'stopped' &&
										!tourState.current.startedExperiment}
									class="w-full h-7.5"
								>
									<span class="flex items-center justify-center gap-1.5 w-full">
										<PlayIcon class="w-3.5 h-3.5 shrink-0" strokeWidth={iconStrokeWidth} />
										Start Training
									</span>
								</ActionButton>
							{:else}
								<div class="grid gap-1 {shouldSuggestRestart ? 'grid-cols-6' : 'grid-cols-4'}">
									<ActionButton color="gray" class="h-7.5 col-span-1" onclick={togglePause}>
										<span class="flex items-center justify-center gap-1.5 w-full">
											{#if trainingState.current === 'paused'}
												<PlayIcon class="w-3.5 h-3.5 shrink-0" strokeWidth={iconStrokeWidth} />
											{:else}
												<PauseIcon class="w-3.5 h-3.5 shrink-0" strokeWidth={iconStrokeWidth} />
											{/if}
										</span>
									</ActionButton>
									<ActionButton color="green" class="h-7.5 col-span-1" onclick={stepForward}>
										<span class="flex items-center justify-center gap-1.5 w-full">
											<StepForwardIcon class="w-3.5 h-3.5 shrink-0" strokeWidth={iconStrokeWidth} />
										</span>
									</ActionButton>

									<ActionButton color="red" class="col-span-1 h-7.5" onclick={stopTraining}>
										<span class="flex items-center justify-center gap-1.5 w-full">
											<!-- Stop -->
											<SquareIcon class="w-3.5 h-3.5 shrink-0" strokeWidth={iconStrokeWidth} />
										</span>
									</ActionButton>
									<ActionButton
										color="blue"
										class={`h-7.5 ${shouldSuggestRestart ? 'col-span-3' : 'col-span-1'}`}
										highlighted={shouldSuggestRestart && !tourState.current.restartedExperiment}
										onclick={restartTraining}
									>
										<span class="flex items-center justify-center gap-1.5 w-full">
											<RefreshCcwIcon class="w-3.5 h-3.5 shrink-0" strokeWidth={iconStrokeWidth} />
											{#if shouldSuggestRestart}
												New Changes
											{/if}
										</span>
									</ActionButton>
								</div>
							{/if}
						</div>
					</div>
					<div class="flex-1 overflow-y-auto min-h-0 relative h-full">
						<Controls />
					</div>
				</div>
			</div>
		{/if}

		<!-- Right Panel: Tabs -->
		{#if shouldShowTabContent}
			<div
				class="relative flex-1 sm:flex-none overflow-hidden overscroll-none flex flex-col h-full @container"
				class:col-span-2={!isMobile.current && configOpen.current === false}
			>
				{#if activeTab.current === 'metrics'}
					<Metrics />
				{:else if activeTab.current === 'about'}
					<About />
				{/if}
			</div>
		{/if}
	</div>
</main>

{#if isDragOver}
	<div
		class="pointer-events-none absolute inset-0 z-40 flex items-center justify-center w-full h-full bg-purple-200/90"
	>
		<div
			class="px-4 py-2 text-purple-700 font-semibold text-xs font-mono tracking-wider uppercase bg-purple-50 border border-dashed border-purple-700"
		>
			Drop checkpoint to start from it
		</div>
	</div>
{/if}

{#if showReplaceDialog}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
		role="button"
		tabindex="0"
		onclick={onBackdropClick}
		onkeydown={(e) => {
			if (e.key === 'Enter' || e.key === ' ') cancelReplace();
		}}
	>
		<div
			class="bg-white p-2 w-[22rem] border border-panel-border-base shadow-lg"
			role="dialog"
			aria-modal="true"
			aria-labelledby="replace-dialog-title"
			tabindex="-1"
		>
			<div
				id="replace-dialog-title"
				class="font-semibold mb-1.5 uppercase font-mono tracking-wider text-xs"
			>
				Replace current run?
			</div>
			<div class="text-sm text-neutral-600 mb-3">
				Starting from this checkpoint will end the current run and begin a new one.
			</div>
			<div class="flex gap-2 justify-end">
				<button
					class="px-1.75 py-0.5 border border-panel-border-base font-mono uppercase tracking-wide text-xs cursor-pointer"
					onclick={cancelReplace}>Cancel</button
				>
				<button
					class="px-1.75 py-0.5 bg-purple-600 text-white font-mono uppercase tracking-wide text-xs cursor-pointer border border-purple-800"
					onclick={confirmReplace}>Replace</button
				>
			</div>
		</div>
	</div>
{/if}

<style>
	/* pin for tailwind: bg-neutral-500 */
	.animate-stepped-pulse {
		animation: stepped-pulse 2s steps(1) infinite;
	}

	@keyframes stepped-pulse {
		0% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}
</style>
