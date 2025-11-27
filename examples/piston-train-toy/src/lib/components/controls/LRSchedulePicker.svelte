<script lang="ts">
	import { config, equalsConfigDefault, resetConfigToDefaults } from '$lib/workspace/config.svelte';
	import {
		ConstantLR,
		CosineAnnealingLR,
		ExponentialLR,
		LinearLR,
		type LRScheduler,
		type Optimizer,
		SequentialLR,
		StepLR
	} from '@piston-ml/piston-web';
	import { MediaQuery } from 'svelte/reactivity';

	import { CheckboxInput, SelectInput, Slider } from 'example-common';

	let logScale = $state(false);
	let chartContainer: HTMLDivElement;
	let containerWidth = $state(300); // Default width
	let stepsToShow = $state(1000); // Local state for visualization range

	const isMobile = new MediaQuery('pointer: coarse');

	// Mock optimizer for scheduler testing
	function createMockOptimizer(learningRate: number) {
		return {
			paramGroups: [{ lr: learningRate, initialLr: learningRate }]
		};
	}

	function generateSchedulerPoints(maxSteps: number = 100) {
		const points: [number, number][] = [];
		const schedulerType = config.optimizer.lrScheduler.type;
		const baseLr = config.optimizer.lr;

		// Create mock optimizer for scheduler testing
		const mockOptimizer = createMockOptimizer(baseLr) as unknown as Optimizer;

		// Create the appropriate scheduler
		let scheduler: LRScheduler<unknown>;
		try {
			switch (schedulerType) {
				case 'constant':
					scheduler = new ConstantLR(
						mockOptimizer,
						config.optimizer.lrScheduler.constantSchedule.factor,
						config.optimizer.lrScheduler.constantSchedule.totalIters
					);
					break;
				case 'linear':
					scheduler = new LinearLR(
						mockOptimizer,
						config.optimizer.lrScheduler.linearSchedule.startFactor,
						config.optimizer.lrScheduler.linearSchedule.endFactor,
						config.optimizer.lrScheduler.linearSchedule.totalIters
					);
					break;
				case 'cosine':
					scheduler = new CosineAnnealingLR(
						mockOptimizer,
						config.optimizer.lrScheduler.cosineAnnealingSchedule.tMax,
						config.optimizer.lrScheduler.cosineAnnealingSchedule.etaMin
					);
					break;
				case 'step':
					scheduler = new StepLR(
						mockOptimizer,
						config.optimizer.lrScheduler.stepSchedule.stepSize,
						config.optimizer.lrScheduler.stepSchedule.gamma
					);
					break;
				case 'exponential':
					scheduler = new ExponentialLR(
						mockOptimizer,
						config.optimizer.lrScheduler.exponentialSchedule.gamma
					);
					break;
				default:
					// If unknown scheduler, just return constant learning rate
					for (let step = 0; step <= maxSteps; step++) {
						points.push([step, baseLr]);
					}
					return points;
			}
		} catch (error) {
			console.warn('Could not create scheduler:', error);
			// Fallback to constant learning rate
			for (let step = 0; step <= maxSteps; step++) {
				points.push([step, baseLr]);
			}
			return points;
		}

		// Wrap scheduler with warmup if enabled
		if (config.optimizer.warmupSteps.present && scheduler) {
			const n = config.optimizer.warmupSteps.value | 0;
			if (n > 0) {
				const warm = new LinearLR(mockOptimizer, 1e-8, 1.0, n);
				scheduler = new SequentialLR(mockOptimizer, [warm, scheduler], [n]);
			}
		}

		// Generate points by stepping through the scheduler
		for (let step = 0; step <= maxSteps; step++) {
			const currentLr = scheduler.getLastLr()[0] || baseLr;
			points.push([step, currentLr]);

			// Step the scheduler for next iteration
			if (step < maxSteps) {
				scheduler.step();
			}
		}

		return points;
	}

	const points = $derived(generateSchedulerPoints(stepsToShow));

	let chartHeight = $state(100);

	// Update chart height when media query or container changes
	$effect(() => {
		// Trigger reactivity when media query changes
		const _isMobile = isMobile.current;

		// Get CSS variable value and convert rem to px
		if (typeof window !== 'undefined' && chartContainer) {
			// Temporarily set a CSS property to convert rem to px
			const originalHeight = chartContainer.style.height;
			chartContainer.style.height = 'calc(var(--spacing) * 32)';
			const computedHeight = parseFloat(getComputedStyle(chartContainer).height);
			chartContainer.style.height = originalHeight;

			chartHeight = computedHeight || 100;
		}
	});

	const chartWidth = $derived(containerWidth || 300);

	// Text measurement elements
	let measurementSvg: SVGSVGElement;
	let leftPadding = $state(30);
	let rightPadding = $state(10);
	let topPadding = $state(15);
	let bottomPadding = $state(20);
	let labelHeight = $state(0);

	const plotWidth = $derived(chartWidth - leftPadding - rightPadding);
	const plotHeight = $derived(chartHeight - topPadding - bottomPadding);

	const svgPoints = $derived.by(() => {
		if (points.length === 0) return '';

		const maxSteps = Math.max(...points.map(([x]) => x));
		const minSteps = Math.min(...points.map(([x]) => x));
		const lrs = points.map(([, y]) => y);
		const maxLr = Math.max(...lrs);
		const minLr = Math.min(...lrs);
		// const minLr = 1e-6; // Fixed bottom at 1e-6

		// Avoid division by zero
		if (maxSteps === minSteps) {
			return '';
		}

		if (maxLr === minLr) {
			// Constant learning rate - draw a horizontal line
			const y = chartHeight - bottomPadding - plotHeight / 2;
			const xStart = leftPadding;
			const xEnd = leftPadding + plotWidth;
			return `${xStart},${y} ${xEnd},${y}`;
		}

		return points
			.map(([step, lr]) => {
				const x = leftPadding + ((step - minSteps) * plotWidth) / (maxSteps - minSteps);

				let y: number;
				if (logScale && minLr > 0 && maxLr > 0) {
					const logMinLr = Math.log10(minLr);
					const logMaxLr = Math.log10(maxLr);
					const logLr = Math.log10(lr);
					y =
						chartHeight - bottomPadding - ((logLr - logMinLr) * plotHeight) / (logMaxLr - logMinLr);
				} else {
					y = chartHeight - bottomPadding - ((lr - minLr) * plotHeight) / (maxLr - minLr);
				}

				return `${x},${y}`;
			})
			.join(' ');
	});

	// Format numbers for display
	function formatNumber(num: number): string {
		// if (num >= 1000) return num.toExponential(2);
		// if (num >= 1) return num.toFixed(3);
		// if (num >= 0.001) return num.toFixed(4);
		return num.toExponential(2);
	}

	const maxLrLabel = $derived.by(() => {
		if (points.length === 0) return '';
		const maxLr = Math.max(...points.map(([, y]) => y));
		return formatNumber(maxLr);
	});

	const minLrLabel = $derived.by(() => {
		if (points.length === 0) return '';
		const minLr = Math.min(...points.map(([, y]) => y));
		return formatNumber(minLr);
	});

	// Function to measure text width
	async function measureTextDimensions(text: string, className: string): Promise<[number, number]> {
		if (!measurementSvg) return [0, 0];

		// Check if fonts are loaded before measuring
		if (document.fonts && document.fonts.ready) {
			await document.fonts.ready;
		}

		const textElement = document.createElementNS('http://www.w3.org/2000/svg', 'text');
		textElement.classList.add(className);
		textElement.textContent = text;

		// eslint-disable-next-line svelte/no-dom-manipulating
		measurementSvg.appendChild(textElement);
		const bbox = textElement.getBBox();
		// eslint-disable-next-line svelte/no-dom-manipulating
		measurementSvg.removeChild(textElement);

		return [bbox.width, bbox.height];
	}

	// Calculate dynamic padding based on text measurements
	$effect(() => {
		const _isMobile = isMobile.current;
		if (!measurementSvg || points.length === 0 || typeof window === 'undefined') return;

		// Wrap this whole thing in an async function so we can wait for fonts to load if they haven't yet
		(async () => {
			// Measure Y-axis labels (LR values)
			const [maxLrWidth] = await measureTextDimensions(maxLrLabel, 'text-2xs');
			const [minLrWidth] = await measureTextDimensions(minLrLabel, 'text-2xs');
			const maxYLabelWidth = Math.max(maxLrWidth, minLrWidth);

			// Measure X-axis label and step values
			const [stepsLabelWidth, stepsLabelHeight] = await measureTextDimensions('steps', 'text-2xs');
			const maxSteps = Math.max(...points.map(([x]) => x));
			const [_maxStepsWidth, maxStepsHeight] = await measureTextDimensions(
				maxSteps.toString(),
				'text-2xs'
			);

			// Calculate maximum height needed for X-axis labels
			const maxXLabelHeight = stepsLabelHeight + maxStepsHeight;

			// Update padding with some buffer
			leftPadding = Math.max(20, maxYLabelWidth + 5);
			// Kind of a dumb way to go about this
			rightPadding = stepsLabelWidth / 2;
			bottomPadding = Math.max(20, maxXLabelHeight); // Dynamic padding based on text height
			labelHeight = stepsLabelHeight;
			topPadding = 15; // Fixed for now
		})();
	});
</script>

<div class="relative w-full space-y-1">
	<!-- Hidden SVG for text measurement -->
	<svg
		bind:this={measurementSvg}
		style="position: absolute; visibility: hidden; width: 0; height: 0;"
		aria-hidden="true"
	></svg>

	<!-- Scheduler Type Selection -->
	<SelectInput
		bind:value={config.optimizer.lrScheduler.type}
		id="lr-scheduler"
		class="w-full"
		options={[
			{ value: 'constant', text: 'Constant' },
			{ value: 'linear', text: 'Linear' },
			{ value: 'cosine', text: 'Cosine Annealing' },
			{ value: 'step', text: 'Step' },
			{ value: 'exponential', text: 'Exponential' }
		]}
		hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.type')}
		onReset={() => resetConfigToDefaults('optimizer.lrScheduler.type')}
	/>

	<!-- Chart -->
	<div class="relative" bind:this={chartContainer} bind:clientWidth={containerWidth}>
		<svg width={chartWidth} height={chartHeight} class="border border-neutral-300 bg-white">
			<!-- Axis lines -->
			<line
				x1={leftPadding}
				y1={chartHeight - bottomPadding}
				x2={chartWidth - rightPadding}
				y2={chartHeight - bottomPadding}
				stroke="#999"
				stroke-width="1"
			/>
			<line
				x1={leftPadding}
				y1={topPadding}
				x2={leftPadding}
				y2={chartHeight - bottomPadding}
				stroke="#999"
				stroke-width="1"
			/>

			<!-- Y-axis labels -->
			{#if points.length > 0}
				{@const lrs = points.map(([, y]) => y)}
				{@const maxLr = Math.max(...lrs)}
				{@const minLr = Math.min(...lrs)}
				{#if maxLr === minLr}
					<!-- Single label for constant learning rate -->
					<text
						x="2"
						y={topPadding + plotHeight / 2 + 3}
						class="text-2xs"
						fill="black"
						text-anchor="start">{maxLrLabel}</text
					>
				{:else}
					<!-- Normal max/min labels -->
					<text x="2" y={topPadding + 4} class="text-2xs" fill="black" text-anchor="start"
						>{maxLrLabel}</text
					>
					<text
						x="2"
						y={chartHeight - bottomPadding - 2}
						class="text-2xs"
						fill="black"
						text-anchor="start">{minLrLabel}</text
					>
				{/if}
			{/if}

			<!-- X-axis label and step number -->
			<text
				x={chartWidth - rightPadding}
				y={chartHeight - 4}
				class="text-2xs"
				fill="black"
				text-anchor="end">steps</text
			>
			{#if points.length > 0}
				<text
					x={chartWidth - rightPadding}
					y={chartHeight - bottomPadding + labelHeight - 1}
					class="text-2xs"
					fill="black"
					text-anchor="end">{Math.max(...points.map(([x]) => x))}</text
				>
			{/if}

			<!-- Schedule curve -->
			{#if svgPoints}
				<polyline points={svgPoints} fill="none" stroke="black" stroke-width="2" />
			{/if}
		</svg>

		<!-- Log scale checkbox -->
		<div class="mt-1">
			<CheckboxInput
				id="log-scale-y-axis"
				label="Log Scale Y-Axis"
				bind:checked={logScale}
				labelClass="ml-1 text-xs"
			/>
		</div>
	</div>

	<!-- Scheduler Parameters -->
	<div class="space-y-1">
		<!-- Steps to show in visualization -->
		<Slider
			id="steps-to-show"
			label="Steps to Show"
			bind:value={stepsToShow}
			min={10}
			max={10_000}
			step={10}
			hasDefaultValue={stepsToShow === 1000}
			onReset={() => (stepsToShow = 1000)}
		/>

		<!-- Linear scheduler parameters -->
		{#if config.optimizer.lrScheduler.type === 'linear'}
			<Slider
				id="lr-scheduler-linear-start-factor"
				label="Start Factor"
				bind:value={config.optimizer.lrScheduler.linearSchedule.startFactor}
				min={0.01}
				max={1}
				step={0.01}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.linearSchedule.startFactor')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.linearSchedule.startFactor')}
			/>
			<Slider
				id="lr-scheduler-linear-end-factor"
				label="End Factor"
				bind:value={config.optimizer.lrScheduler.linearSchedule.endFactor}
				min={0.01}
				max={1}
				step={0.01}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.linearSchedule.endFactor')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.linearSchedule.endFactor')}
			/>
			<Slider
				id="lr-scheduler-linear-total-iters"
				label="Total Iterations"
				bind:value={config.optimizer.lrScheduler.linearSchedule.totalIters}
				min={1}
				max={1000}
				step={1}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.linearSchedule.totalIters')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.linearSchedule.totalIters')}
			/>
		{/if}

		<!-- Constant scheduler parameters -->
		{#if config.optimizer.lrScheduler.type === 'constant'}
			<Slider
				id="lr-scheduler-constant-factor"
				label="Factor"
				bind:value={config.optimizer.lrScheduler.constantSchedule.factor}
				min={0.01}
				max={2}
				step={0.01}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.constantSchedule.factor')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.constantSchedule.factor')}
			/>
			<Slider
				id="lr-scheduler-constant-total-iters"
				label="Total Iterations"
				bind:value={config.optimizer.lrScheduler.constantSchedule.totalIters}
				min={1}
				max={1000}
				step={1}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.constantSchedule.totalIters')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.constantSchedule.totalIters')}
			/>
		{/if}

		<!-- Cosine annealing scheduler parameters -->
		{#if config.optimizer.lrScheduler.type === 'cosine'}
			<Slider
				id="lr-scheduler-cosine-annealing-t-max"
				label="Maximum Iterations"
				bind:value={config.optimizer.lrScheduler.cosineAnnealingSchedule.tMax}
				min={1}
				max={10_000}
				step={1}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.cosineAnnealingSchedule.tMax')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.cosineAnnealingSchedule.tMax')}
			/>
			<Slider
				id="lr-scheduler-cosine-annealing-eta-min"
				label="Minimum Learning Rate"
				bind:value={config.optimizer.lrScheduler.cosineAnnealingSchedule.etaMin}
				min={1e-5}
				max={1e-2}
				step={1e-5}
				useLog
				hasDefaultValue={equalsConfigDefault(
					'optimizer.lrScheduler.cosineAnnealingSchedule.etaMin'
				)}
				onReset={() =>
					resetConfigToDefaults('optimizer.lrScheduler.cosineAnnealingSchedule.etaMin')}
			/>
		{/if}

		<!-- Step scheduler parameters -->
		{#if config.optimizer.lrScheduler.type === 'step'}
			<Slider
				id="lr-scheduler-step-step-size"
				label="Step Size"
				bind:value={config.optimizer.lrScheduler.stepSchedule.stepSize}
				min={1}
				max={1000}
				step={1}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.stepSchedule.stepSize')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.stepSchedule.stepSize')}
			/>
			<Slider
				id="lr-scheduler-step-gamma"
				label="Gamma"
				bind:value={config.optimizer.lrScheduler.stepSchedule.gamma}
				min={0.01}
				max={1}
				step={0.01}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.stepSchedule.gamma')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.stepSchedule.gamma')}
			/>
		{/if}

		<!-- Exponential scheduler parameters -->
		{#if config.optimizer.lrScheduler.type === 'exponential'}
			<Slider
				id="lr-scheduler-exponential-gamma"
				label="Gamma"
				bind:value={config.optimizer.lrScheduler.exponentialSchedule.gamma}
				min={0.8}
				max={0.999}
				step={0.001}
				hasDefaultValue={equalsConfigDefault('optimizer.lrScheduler.exponentialSchedule.gamma')}
				onReset={() => resetConfigToDefaults('optimizer.lrScheduler.exponentialSchedule.gamma')}
			/>
		{/if}
	</div>
</div>
