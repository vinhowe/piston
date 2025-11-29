<script lang="ts">
	import type { ValidationStep } from '$lib/train/validation';
	import type { MetricData } from '$lib/workspace/runs.svelte';
	import type { CallbackDataParams } from 'echarts/types/dist/shared';

	import createEChartsAttachment, {
		extractStepFromAxisPointerEvent,
		nearestIndex,
		publishClear,
		publishMove,
		subscribe
	} from '$lib/attachments/echarts.svelte';
	import { decodeSingle, PreTrainedTokenizer } from '$lib/train/tokenizer';
	import { getLatestRunDataset } from '$lib/workspace/config.svelte';
	import { getLatestRun, runsMap } from '$lib/workspace/runs.svelte';
	import { maxCompletions } from '$lib/workspace/ui.svelte';
	import { BarChart, HeatmapChart } from 'echarts/charts';
	import { GridComponent, TooltipComponent, VisualMapComponent } from 'echarts/components';
	import * as echarts from 'echarts/core';

	import { ResetValueButton } from 'example-common';
	import CompletionsToken from './CompletionsToken.svelte';

	echarts.use([VisualMapComponent, GridComponent, TooltipComponent, HeatmapChart, BarChart]);

	const completionCountOptions = [1, 2, 4, 8, 16, 24, 32] as const;

	const { tokenizer } = $derived(getLatestRunDataset() ?? { tokenizer: null });
	let selectedProbsStep: { categories: string[]; data: number[]; step: number } | null =
		$state(null);
	let activeStepCol: number | null = $state(null);
	let activeStep: number | null = $state(null);

	// Tokenizer for synchronous tooltip decoding
	let tooltipTokenizer: PreTrainedTokenizer | null = $state(null);
	const heatmapChartId: string = crypto.randomUUID();

	$effect(() => {
		if (tokenizer instanceof Promise) {
			tokenizer
				.then((t) => {
					tooltipTokenizer = t;
				})
				.catch(() => {
					tooltipTokenizer = null;
				});
		} else {
			tooltipTokenizer = tokenizer;
		}
	});

	const completionsData = $derived.by(() => {
		const latestRunId = getLatestRun()?.runId;
		if (!latestRunId) {
			return null;
		}

		const latestRun = runsMap.get(latestRunId);

		if (!latestRun) {
			return null;
		}

		// Find val generation metrics
		let valMetric: MetricData | undefined = latestRun.metrics.get('validation/completions');

		if (!valMetric || valMetric.data.length === 0) {
			return null;
		}

		// Filter to get only ValStep data
		const valSteps = valMetric.data.filter(
			(step): step is ValidationStep => 'type' in step && step.type === 'validation'
		);

		if (valSteps.length === 0) {
			return null;
		}

		// Get the latest val step
		const targetStep =
			selectedProbsStep || activeStep
				? (valSteps.find(
						(step) => step.step === activeStep || step.step === selectedProbsStep?.step
					) ?? valSteps[valSteps.length - 1])
				: valSteps[valSteps.length - 1];

		// Get all step numbers
		const stepNumbers = valSteps.map((step) => step.step);

		// Find targets from the first step (they're only logged once)
		let targets: number[][] | undefined;
		for (const step of valSteps) {
			if (step.targets) {
				targets = step.targets;
				break;
			}
		}

		// Do not return null if there are no targets; matches-only mode is allowed
		return {
			targetStep,
			valSteps,
			stepNumbers,
			targets,
			stepNumber: targetStep.step,
			encoderInputs: targetStep.encoderInputs,
			decoderPromptLengths: targetStep.decoderPromptLengths
		};
	});

	const visibleCompletions = $derived(
		completionsData?.targetStep.completions.slice(0, Math.min(+maxCompletions.current, 50)) ?? []
	);

	// If we have explicit match data, we can judge tokens (green/red). Otherwise, treat as free completion.
	const hasMatchData = $derived(
		Boolean(completionsData?.targetStep.matches && completionsData.targetStep.matches.length > 0)
	);

	// Compare tokens (skip masked -100 targets) and produce render info
	function compareTokenIds(
		generatedIds: number[],
		targetIds: number[],
		tokenizer: PreTrainedTokenizer | null
	): Array<
		| { kind: 'prompt'; promptText: string }
		| { kind: 'compare'; isCorrect: boolean; targetText: string; genText: string }
	> {
		if (!tokenizer) {
			throw new Error('Attempted to compare token IDs without a tokenizer for the latest run');
		}

		const shift = 1;

		const results: Array<
			| { kind: 'prompt'; promptText: string }
			| { kind: 'compare'; isCorrect: boolean; targetText: string; genText: string }
		> = [];
		const maxLength = Math.max(generatedIds.length, targetIds.length);
		for (let i = 0; i < maxLength; i++) {
			const targetId = targetIds[i];
			const genAlignedId = generatedIds[i + shift];
			if (targetId === -100) {
				if (genAlignedId !== undefined) {
					results.push({ kind: 'prompt', promptText: decodeSingle(genAlignedId, tokenizer) });
				}
				continue;
			}
			if (targetId === undefined) {
				continue;
			}
			const isCorrect = genAlignedId !== undefined && genAlignedId === targetId;
			const targetText = decodeSingle(targetId, tokenizer);
			const genText = genAlignedId !== undefined ? decodeSingle(genAlignedId, tokenizer) : '';
			results.push({ kind: 'compare', isCorrect, targetText, genText });
		}
		return results;
	}

	function visualizeToken(text: string): string {
		return text.replace(/\s/g, '␣');
	}

	function escapeHtml(text: string): string {
		return text
			.replace(/&/g, '&amp;')
			.replace(/</g, '&lt;')
			.replace(/>/g, '&gt;')
			.replace(/"/g, '&quot;')
			.replace(/'/g, '&#39;');
	}

	// Hovered and selected token focus across examples
	let hoveredFocus: { exampleIndex: number; tokenIndex: number } | null = $state(null);
	let lastRunId: string | null = $state(null);

	$effect(() => {
		// Detect run change and reset selection
		const currentRunId = getLatestRun()?.runId ?? null;
		if (lastRunId !== currentRunId) {
			lastRunId = currentRunId;
			hoveredFocus = null;
			selectedProbsStep = null;
		}
	});

	const MAX_PROBS = 16;
	const MAX_STEPS = 64;

	const chartOptions = $derived.by(() => {
		if (!completionsData) return null;

		// Determine current focus (example + token step index)
		const focus = hoveredFocus;
		const defaultExample = 0;
		const exampleIndex = Math.max(
			0,
			Math.min(
				(completionsData.targetStep.completions.length ?? 1) - 1,
				focus?.exampleIndex ?? defaultExample
			)
		);
		// Compute prefix length for mapping tokenIndex->probIndex
		const defaultPrefixLen = 1;
		const prefixLen = completionsData.decoderPromptLengths?.[exampleIndex] ?? defaultPrefixLen;
		const requestedTokenIndex = focus?.tokenIndex ?? null;
		const requestedProbIndex =
			requestedTokenIndex != null ? Math.max(0, requestedTokenIndex - prefixLen) : null;

		const steps = completionsData.valSteps.length;
		// For each validation step, pick the requested prob row from the focused example,
		// falling back to that example's last available row.
		const stepRows: Array<number[] | null> = new Array(steps).fill(null);
		for (let s = 0; s < steps; s++) {
			const stepData = completionsData.valSteps[s];
			const completion = stepData.completions[exampleIndex];
			const rows = completion.probs;
			if (rows.length > 0) {
				const idx =
					requestedProbIndex != null
						? Math.min(rows.length - 1, requestedProbIndex)
						: rows.length - 1;
				stepRows[s] = rows[idx] ?? null;
			}
		}
		let vocabSize = 0;
		for (let s = 0; s < steps; s++) {
			if (stepRows[s]) {
				vocabSize = (stepRows[s] as number[]).length;
				break;
			}
		}
		if (vocabSize === 0) {
			return null;
		}

		// Determine token order from the MOST RECENT step's last-available probs
		let baseStepIndex = -1;
		for (let s = steps - 1; s >= 0; s--) {
			if (stepRows[s]) {
				baseStepIndex = s;
				break;
			}
		}
		if (baseStepIndex === -1) return null;
		const baseRow = stepRows[baseStepIndex] as number[];
		const descOrder = Array.from({ length: vocabSize }, (_, i) => i).sort(
			(a, b) => baseRow[b] - baseRow[a]
		);
		const topIds = descOrder.slice(0, Math.min(MAX_PROBS, vocabSize));
		let selectedTokenIds: number[] = topIds;

		// Select up to MAX_STEPS steps using exponential sampling (denser near most recent)
		const totalSteps = steps;
		const k = Math.min(totalSteps, MAX_STEPS);
		let selectedStepIndices: number[];
		if (k === totalSteps) {
			selectedStepIndices = Array.from({ length: totalSteps }, (_, i) => i);
		} else if (k <= 1) {
			selectedStepIndices = [totalSteps - 1];
		} else {
			// eslint-disable-next-line svelte/prefer-svelte-reactivity
			const exponentialChosen = new Set<number>();
			for (let i = 0; i < k; i++) {
				const t = i / (k - 1);
				const tPrime = 1 - t; // bias selection toward more recent steps
				const idxFromStart = Math.floor(Math.pow(totalSteps, tPrime) - 1);
				let idx = totalSteps - 1 - idxFromStart;
				if (idx < 0) idx = 0;
				if (idx >= totalSteps) idx = totalSteps - 1;
				exponentialChosen.add(idx);
			}
			selectedStepIndices = Array.from(exponentialChosen).sort((a, b) => a - b);
			// Ensure we have exactly k unique indices by backfilling from the end
			let backfill = totalSteps - 1;
			while (selectedStepIndices.length < k && backfill >= 0) {
				if (!selectedStepIndices.includes(backfill)) selectedStepIndices.push(backfill);
				backfill--;
			}
			selectedStepIndices.sort((a, b) => a - b);
		}

		// Prepare axes for selected steps only
		const xData = selectedStepIndices.map((i) => completionsData.stepNumbers[i]);
		const yData = selectedTokenIds.map((id) => String(id));

		// Build heatmap data: log-softmax per step
		let minVal = Infinity;
		let maxVal = -Infinity;
		let sum = 0;
		let sumSq = 0;
		let count = 0;
		const triples: Array<[number, number, number]> = [];
		for (let xi = 0; xi < selectedStepIndices.length; xi++) {
			const s = selectedStepIndices[xi];
			const row = stepRows[s];
			if (!row) continue;
			let max = -Infinity;
			for (let vi = 0; vi < vocabSize; vi++) {
				const v = row[vi];
				if (v > max) max = v;
			}
			let sumExp = 0;
			for (let vi = 0; vi < vocabSize; vi++) {
				sumExp += Math.exp(row[vi] - max);
			}
			const logDen = Math.log(sumExp) + max;
			for (let yi = 0; yi < selectedTokenIds.length; yi++) {
				const tokenId = selectedTokenIds[yi];
				const val = row[tokenId] - logDen; // log-prob
				if (val < minVal) minVal = val;
				if (val > maxVal) maxVal = val;
				sum += val;
				sumSq += val * val;
				count++;
				triples.push([xi, yi, val]);
			}
		}

		// Compute mean and standard deviation across displayed cells
		const mean = count > 0 ? sum / count : 0;
		const variance = count > 0 ? Math.max(sumSq / count - mean * mean, 0) : 0;
		const std = Math.sqrt(variance);
		const sigma = 2;
		const vmin = std > 0 ? Math.max(minVal, mean - sigma * std) : minVal;
		const vmax = std > 0 ? Math.min(maxVal, mean + sigma * std) : maxVal;

		let titleText = 'probs';
		if (vocabSize > MAX_PROBS) {
			titleText = `probs (top ${MAX_PROBS})`;
		}

		const tokenCategories = yData;
		// Build per-step, independently-sorted top-K token ids and corresponding bar values
		const topK = tokenCategories.length;
		const tokenCategoriesByCol: string[][] = [];
		const tokensByCol: number[][] = [];
		for (let xi = 0; xi < selectedStepIndices.length; xi++) {
			const s = selectedStepIndices[xi];
			const row = stepRows[s];
			if (row) {
				const order = Array.from({ length: vocabSize }, (_, i) => i).sort(
					(a, b) => row[b] - row[a]
				);
				const ids = order.slice(0, topK);
				tokenCategoriesByCol.push(ids.map((id) => String(id)));
				tokensByCol.push(ids.map((id) => row[id] ?? NaN));
			} else {
				// Fallback to base categories with zero values if row is missing
				tokenCategoriesByCol.push([...tokenCategories]);
				tokensByCol.push(new Array(topK).fill(0));
			}
		}

		const initialCol = Math.max(0, selectedStepIndices.length - 1);
		const initialBars = tokensByCol[initialCol] ?? new Array(tokenCategories.length).fill(0);

		const heatmapOption = {
			animation: false,
			transitionDuration: 0,
			border: 'none',
			backgroundColor: 'transparent',
			tooltip: {
				animation: false,
				transitionDuration: 0,
				trigger: 'axis',
				borderColor: '#ccc',
				borderWidth: 1,
				borderRadius: 0,
				textStyle: {
					color: 'black',
					fontSize: 11
				},
				padding: [6, 8],
				formatter: function (params: CallbackDataParams[]) {
					if (!params || !params.length) {
						return '';
					}

					return `<div class="leading-none flex flex-col gap-1"><div class="font-mono font-bold">Step ${params[0].name}</div>`;
				}
			},
			xAxis: {
				type: 'category',
				data: xData,
				name: 'Step',
				nameLocation: 'middle',
				nameGap: 20
			},
			yAxis: {
				type: 'category',
				data: yData,
				inverse: true,
				name: 'Token ID',
				nameLocation: 'middle',
				axisLabel: { show: false }
			},
			grid: {
				left: '5%',
				right: '5%',
				bottom: '10%',
				top: '5%',
				containLabel: true
			},
			visualMap: {
				show: false,
				min: vmin,
				max: vmax,
				realtime: false,
				itemWidth: 10,
				itemHeight: '80',
				padding: 3,
				left: 'center',
				orient: 'horizontal'
			},
			series: [
				{
					id: 'probs-heatmap',
					type: 'heatmap',
					data: triples,
					animation: false
				}
			]
		};

		const barOption = {
			animation: false,
			transitionDuration: 0,
			backgroundColor: 'transparent',
			tooltip: {
				animation: false,
				transitionDuration: 0,
				trigger: 'axis',
				axisPointer: { type: 'shadow' },
				borderColor: '#ccc',
				borderWidth: 1,
				borderRadius: 0,
				textStyle: { color: 'black', fontSize: 11 },
				padding: [6, 8],
				formatter: function (params: CallbackDataParams[]) {
					if (!params || !params.length) {
						return '';
					}
					const firstParam = params[0];
					const tokenIdStr = String(firstParam.name);
					const tokenIdNum = Number(tokenIdStr);
					const tokenText = visualizeToken(decodeSingle(tokenIdNum, tooltipTokenizer));
					const safeTokenText = escapeHtml(tokenText);
					const raw = firstParam.value;
					const val = Array.isArray(raw)
						? Number(raw[raw.length - 1] as number)
						: Number(raw as number);
					const valText = Number.isFinite(val) ? val.toFixed(4) : String(raw);
					return (
						`<div class="leading-none flex flex-col gap-1">` +
						`<div class="font-mono font-bold">${safeTokenText}</div>` +
						`<div class="font-mono text-neutral-600">Prob: ${valText}</div>` +
						`</div>`
					);
				}
			},
			title: {
				text: titleText,
				left: 'center',
				textStyle: {
					fontSize: 11
				},
				itemGap: 0,
				top: 2
			},
			grid: {
				left: '5%',
				right: '5%',
				top: '40%',
				bottom: '10%',
				containLabel: false
			},
			xAxis: [
				{
					type: 'category',
					data: tokenCategories,
					axisPointer: { show: true, type: 'shadow' },
					triggerEvent: true,
					axisTick: { show: false },
					axisLabel: { show: false },
					axisLine: { show: false }
				}
			],
			yAxis: [
				{
					type: 'value',
					axisPointer: { show: false },
					axisTick: { show: false },
					axisLabel: { show: false },
					splitLine: { show: false },
					axisLine: { show: false }
				}
			],
			series: [
				{
					id: 'token-bars',
					name: 'token probs',
					type: 'bar',
					barWidth: '99%',
					label: { show: false },
					data: initialBars
				}
			]
		};

		return {
			heatmap: heatmapOption,
			bar: barOption,
			tokensByCol,
			tokenCategoriesByCol,
			stepAxisData: xData
		};
	});

	const createHeatmapAttachment = () =>
		createEChartsAttachment({
			opts: () => (chartOptions ? chartOptions.heatmap : null),
			setup: (chart) => {
				const getDisplayedSteps = (): number[] => chartOptions?.stepAxisData ?? [];
				const getTopKForCol = (xi: number): number => {
					const byCol = chartOptions?.tokensByCol ?? [];
					if (xi >= 0 && xi < byCol.length) return byCol[xi]?.length ?? 0;
					return byCol[0]?.length ?? 0;
				};

				chart.on('updateAxisPointer', (evt) => {
					const step = extractStepFromAxisPointerEvent(evt);
					if (!Number.isFinite(step)) return;
					const steps = getDisplayedSteps();
					const last = Math.max(0, steps.length - 1);
					const xi = Math.max(0, Math.min(last, Math.floor(step ?? 0)));
					activeStepCol = xi;
					activeStep = steps[xi];
					const s = activeStep;
					const runId = getLatestRun()?.runId;
					if (runId != null && Number.isFinite(s)) {
						publishMove({ sourceId: heatmapChartId, runId, step: s });
					}
				});

				chart.on(
					'click',
					(evt: { data?: unknown; value?: unknown; dataIndex?: number } | undefined) => {
						let xi: number | null = null;
						if (evt && Array.isArray(evt.data)) {
							const first = evt.data[0];
							if (typeof first === 'number') xi = first;
						} else if (evt && Array.isArray(evt.value)) {
							const first = evt.value[0];
							if (typeof first === 'number') xi = first;
						} else if (evt && typeof evt.dataIndex === 'number') {
							xi = evt.dataIndex;
						}
						if (xi !== null && chartOptions) {
							const tbc = chartOptions.tokensByCol;
							const catsByCol = chartOptions.tokenCategoriesByCol;
							const stepAxis = chartOptions.stepAxisData;
							const last = Math.max(0, tbc.length - 1);
							const col = Math.max(0, Math.min(last, Math.floor(xi)));
							const stepValue = stepAxis[col];
							selectedProbsStep = {
								categories: catsByCol[col],
								data: tbc[col],
								step: stepValue
							};
						}
					}
				);

				chart.on('globalout', () => {
					if (activeStep === null) return;
					activeStepCol = null;
					activeStep = null;
					publishClear({ sourceId: heatmapChartId });
				});

				const unsubscribe = subscribe(
					({ sourceId, runId, step }) => {
						if (sourceId === heatmapChartId) return;
						const myRunId = getLatestRun()?.runId;
						if (!myRunId || myRunId !== runId) return;
						const steps = getDisplayedSteps();
						if (steps.length === 0) return;
						const first = steps[0];
						const last = steps[steps.length - 1];
						if (step < first || step > last) return;
						const xi = nearestIndex(steps, step);
						if (xi < 0) return;
						const topK = getTopKForCol(xi);
						const dataIndex = Math.max(0, xi) * Math.max(1, topK);
						chart.dispatchAction(
							{ type: 'updateAxisPointer', seriesIndex: 0, dataIndex },
							{ silent: true }
						);
						if (step === activeStep) return;
						activeStepCol = xi;
						activeStep = steps[xi];
					},
					({ sourceId }) => {
						if (sourceId === heatmapChartId) return;
						// ECharts accepts a second options argument with { silent: true }
						chart.dispatchAction({ type: 'hideTip' }, { silent: true });
					}
				);

				return () => unsubscribe();
			}
		});
</script>

<div class="w-full p-2 md:p-3 h-[250px] flex flex-col">
	<div class="mb-4">
		<span class="text-sm text-neutral-800">
			<span class="font-bold text-base text-neutral-700">validation/completions</span>
			{#if completionsData}
				<select
					class="align-middle border border-neutral-200 text-xs bg-white -translate-y-px ml-2"
					bind:value={maxCompletions.current}
					aria-label="Number of completions to show"
				>
					{#each completionCountOptions as count (count)}
						<option value={count}>{count}</option>
					{/each}
				</select>
				of {completionsData.targetStep.completions.length}
			{/if}
		</span>
		{#if completionsData}
			<p class="text-sm text-neutral-600">
				Step {completionsData.stepNumber} • Temp: {completionsData.targetStep.samplingParams
					.temperature}
			</p>
		{/if}
	</div>

	{#if !completionsData}
		<div class="flex items-center justify-center h-20 text-neutral-500 text-sm">
			<p>No validation data available. Validation will run during training.</p>
		</div>
	{:else}
		{@const visibleCompletionsNumberWidth = visibleCompletions.length.toString().length}
		{@const focus = hoveredFocus}
		<div class="flex-1 min-h-0 flex">
			<div
				class="overflow-y-auto flex flex-wrap gap-1 md:gap-2 flex-1 min-h-0 -mb-2 md:-mb-3 pb-2 md:pb-3 items-start justify-start"
				role="list"
			>
				{#await tokenizer then tokenizer}
					{#each visibleCompletions as completion, index (index)}
						{@const genIds = completion.tokenIds}
						{@const hasTargets = Boolean(completionsData.targets)}
						{@const tgtIds = hasTargets ? completionsData.targets?.[index] || [] : []}
						{@const tokenComparisons = hasTargets ? compareTokenIds(genIds, tgtIds, tokenizer) : []}
						{@const matchesRow = completionsData.targetStep.matches?.[index] || []}
						{@const prefixLen = completionsData.decoderPromptLengths?.[index] ?? 1}

						<div
							role="button"
							tabindex="0"
							class={`border cursor-pointer border-neutral-200 p-2 md:p-3 gap-x-2 flex ${focus?.exampleIndex === index ? 'ring-2 ring-blue-300 ring-inset' : ''}`}
							onmouseenter={() =>
								(hoveredFocus = {
									exampleIndex: index,
									tokenIndex: hoveredFocus?.tokenIndex ?? 0
								})}
							onmouseleave={() => (hoveredFocus = null)}
						>
							<div class="text-sm tracking-wider font-mono uppercase font-bold md:mr-1">
								{(index + 1).toString().padStart(visibleCompletionsNumberWidth, '\u00A0')}
							</div>

							<!-- Tokens -->
							<div class="font-mono text-sm flex items-end -space-x-px w-max">
								{#if hasTargets}
									{#each tokenComparisons as item, tIndex (tIndex)}
										{@const isPromptItem = item.kind === 'prompt'}
										{@const completedBefore = tokenComparisons
											.slice(0, tIndex)
											.filter((it) => it.kind !== 'prompt').length}
										{@const sequenceTokenIndex = isPromptItem
											? prefixLen
											: prefixLen + Math.max(0, completedBefore)}
										{@const isHighlighted =
											!isPromptItem &&
											focus?.exampleIndex === index &&
											focus?.tokenIndex === sequenceTokenIndex}
										{#if item.kind === 'prompt'}
											<CompletionsToken
												actualText={item.promptText}
												targetText={item.promptText}
												variant="prompt"
												highlighted={false}
												exampleIndex={index}
												tokenIndex={sequenceTokenIndex}
												disabled={true}
												onHover={() => {}}
												onLeave={() => {}}
												onSelect={() => {}}
											/>
										{:else if item.isCorrect}
											<CompletionsToken
												actualText={item.genText}
												targetText={item.genText}
												variant="correct"
												highlighted={isHighlighted}
												exampleIndex={index}
												tokenIndex={sequenceTokenIndex}
												onHover={(ei, ti) => (hoveredFocus = { exampleIndex: ei, tokenIndex: ti })}
												onLeave={() => (hoveredFocus = null)}
											/>
										{:else}
											<CompletionsToken
												actualText={item.genText}
												targetText={item.targetText}
												variant="incorrect"
												highlighted={isHighlighted}
												exampleIndex={index}
												tokenIndex={sequenceTokenIndex}
												onHover={(ei, ti) => (hoveredFocus = { exampleIndex: ei, tokenIndex: ti })}
												onLeave={() => (hoveredFocus = null)}
											/>
										{/if}
									{/each}
								{:else if genIds.length === 0}
									<span class="text-neutral-400 italic">[empty]</span>
								{:else}
									{#each genIds as id, tIndex (tIndex)}
										{@const text = decodeSingle(id, tokenizer)}
										{@const isPrompt = tIndex < prefixLen}
										{@const match = matchesRow[tIndex - prefixLen]}
										{@const variant = isPrompt
											? 'prompt'
											: !hasMatchData || matchesRow.length === 0
												? 'generated'
												: match === true
													? 'correct'
													: match === false
														? 'incorrect'
														: 'neutral'}
										{@const isHighlighted =
											!isPrompt && focus?.exampleIndex === index && focus?.tokenIndex === tIndex}
										<CompletionsToken
											actualText={text}
											{variant}
											highlighted={isHighlighted}
											exampleIndex={index}
											tokenIndex={tIndex}
											disabled={isPrompt}
											onHover={(ei, ti) => (hoveredFocus = { exampleIndex: ei, tokenIndex: ti })}
											onLeave={() => (hoveredFocus = null)}
										/>
									{/each}
								{/if}
							</div>
						</div>
					{/each}
				{/await}
			</div>

			<!-- Token probs bar chart -->
			{#if chartOptions}
				<!-- Probs heatmap -->
				<div class="min-h-0 w-30 sm:w-35 md:w-45 flex flex-col bg-white border border-neutral-200">
					<div class="relative h-18 w-full">
						{#if selectedProbsStep}
							<div class="absolute top-1 right-1 z-10">
								<ResetValueButton
									hasDefaultValue={false}
									onReset={() => (selectedProbsStep = null)}
								/>
							</div>
						{/if}
						<div
							class="h-full w-full"
							{@attach createEChartsAttachment({
								opts: () => {
									if (!chartOptions) return null;
									const baseBar = chartOptions.bar;
									const tbc = chartOptions.tokensByCol ?? [];
									const last = Math.max(0, tbc.length - 1);
									const col =
										activeStepCol == null
											? last
											: Math.max(0, Math.min(last, Math.floor(activeStepCol)));
									const defaultCategories = chartOptions.tokenCategoriesByCol[col];
									const defaultData = tbc[col];
									const categories = selectedProbsStep
										? selectedProbsStep.categories
										: defaultCategories;
									const data = selectedProbsStep ? selectedProbsStep.data : defaultData;
									return {
										...baseBar,
										title: selectedProbsStep
											? { ...baseBar.title, text: `probs (step ${selectedProbsStep.step})` }
											: baseBar?.title,
										xAxis: [
											{
												type: 'category',
												data: categories,
												axisPointer: { show: true, type: 'shadow' },
												triggerEvent: true,
												axisTick: { show: false },
												axisLabel: { show: false },
												axisLine: { show: false }
											}
										],
										series: [{ id: 'token-bars', type: 'bar', data }]
									};
								}
							})}
						></div>
					</div>
					<div class="h-full w-full" {@attach createHeatmapAttachment()}></div>
				</div>
			{/if}
		</div>
	{/if}
</div>
