<script lang="ts">
	import ActionButton from '$lib/components/ActionButton.svelte';
	import Controls from '$lib/components/controls/Controls.svelte';
	import { config, initSharedConfigUrlSync } from '$lib/workspace/config.svelte';
	import { currentRun, resetWorkspace, runCounter } from '$lib/workspace/runs.svelte';
	import {
		activeTab,
		configOpen,
		getIconStrokeWidth,
		hasWebGPU,
		isMobile,
		restartTraining,
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
		trainingState,
		workerReady
	} from '$lib/workspace/workers.svelte';
	import {
		ChartLine,
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
		initializeWorkers();

		const uiCleanup = setupUI();

		return () => {
			uiCleanup();
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
						</div>
					{/if}
					<div class="p-1 border-b border-panel-border-base shrink-0">
						<div class="space-y-1">
							{#if trainingState.current === 'stopped'}
								<ActionButton
									color="green"
									disabled={trainingState.current !== 'stopped' || !workerReady.current}
									onclick={startTraining}
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
