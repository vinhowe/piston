<script lang="ts">
	import { onMount } from 'svelte';

	import FormLabel from './FormLabel.svelte';
	import ResetValueButton from './ResetValueButton.svelte';

	let {
		min,
		max,
		useLog = false,
		base = 10,
		showTicks = true,
		tickCount = 10,
		step = 0.01,
		debounceMs = 120,
		label = '',
		id,
		value = $bindable(10),
		unit,
		tickFormatter,
		hasDefaultValue = false,
		onReset = undefined,
		class: className
	}: {
		min: number;
		max: number;
		useLog?: boolean;
		base?: number;
		showTicks?: boolean;
		tickCount?: number;
		step?: number;
		debounceMs?: number;
		label?: string;
		id: string;
		value?: number;
		unit?: string;
		tickFormatter?: (value: number) => string;
		hasDefaultValue?: boolean;
		onReset?: () => void;
		class?: string;
	} = $props();

	let sliderPosition = $state(0);
	let isDragging = $state(false);
	let trackRef: HTMLDivElement | null = $state(null);
	let thumbRef: HTMLDivElement | null = $state(null);
	let inputValueString = $state('');
	let isInputFocused = $state(false);
	let lastValidNumericValue = $state(value);
	let displayValue = $state(value);
	let commitTimeout: number | null = $state(null);

	const EPSILON = 1e-8;
	const EXPONENTIAL_PRECISION = 2;

	// Helper for custom base logarithm
	function customLog(val: number, base: number): number {
		// Fallback to natural log for invalid bases
		if (base <= 0 || base === 1) return Math.log(val);
		return Math.log(val) / Math.log(base);
	}

	// Helper to format exponential notation without trailing zeros in the mantissa
	function formatExponential(num: number, precision?: number): string {
		const exponentialString =
			precision === undefined ? num.toExponential() : num.toExponential(precision);
		const parts = exponentialString.split('e');
		if (parts.length === 2) {
			let mantissa = parts[0];
			if (mantissa.includes('.')) {
				while (mantissa.endsWith('0')) {
					mantissa = mantissa.slice(0, -1);
				}
				if (mantissa.endsWith('.')) {
					mantissa = mantissa.slice(0, -1);
				}
			}
			return `${mantissa}e${parts[1]}`;
		}
		return exponentialString; // Fallback if not in expected format
	}

	// Helper to format the input value string
	function getFormattedInputValue(
		currentValue: number,
		isLog: boolean,
		currentStep: number,
		{ shortExponential = true }: { shortExponential?: boolean } = {}
	): string {
		if (isLog) {
			return shortExponential
				? formatExponential(currentValue, EXPONENTIAL_PRECISION)
				: currentValue.toExponential(EXPONENTIAL_PRECISION);
		} else if (currentStep > 0 && currentStep % 1 === 0) {
			// Check if step is an integer
			return currentValue.toFixed(0);
		} else {
			const decimals =
				currentStep > 0 && currentStep < 1
					? Math.ceil(-Math.log10(currentStep))
					: currentValue < 1
						? 3
						: 2;
			return currentValue.toFixed(decimals);
		}
	}

	$effect(() => {
		const currentMinForLog = Math.max(EPSILON, min);
		const currentValueForLog = Math.max(currentMinForLog, displayValue);
		let calculatedPosition = 0;

		if (useLog) {
			const actualLogBase = base > 1 ? base : Math.E;
			const minL = customLog(currentMinForLog, actualLogBase);
			const maxL = customLog(max, actualLogBase);
			if (maxL > minL && max > 0 && currentMinForLog > 0) {
				const valueL = customLog(currentValueForLog, actualLogBase);
				calculatedPosition = ((valueL - minL) / (maxL - minL)) * 100;
			} else {
				calculatedPosition = 0;
			}
		} else {
			const range = max - min;
			if (range > 0) {
				calculatedPosition = ((displayValue - min) / range) * 100;
			} else {
				calculatedPosition = 0;
			}
		}
		sliderPosition = Math.max(0, Math.min(100, calculatedPosition));
	});

	$effect(() => {
		if (!isInputFocused) {
			inputValueString = getFormattedInputValue(displayValue, useLog, step);
		}
		if (Math.abs(displayValue - lastValidNumericValue) > EPSILON / 100) {
			lastValidNumericValue = displayValue;
		}
	});

	// Keep displayValue in sync with external value when not editing
	$effect(() => {
		if (!isDragging && !isInputFocused) {
			displayValue = value;
		}
	});

	onMount(() => {
		lastValidNumericValue = displayValue;
		inputValueString = getFormattedInputValue(displayValue, useLog, step);
	});

	function clampToRange(v: number): number {
		return Math.max(min, Math.min(max, v));
	}

	function clearCommitTimeout() {
		if (commitTimeout !== null) {
			clearTimeout(commitTimeout);
			commitTimeout = null;
		}
	}

	function scheduleCommit(nextValue: number) {
		displayValue = nextValue;
		clearCommitTimeout();
		commitTimeout = window.setTimeout(() => {
			value = clampToRange(displayValue);
			commitTimeout = null;
		}, debounceMs);
	}

	function flushCommit() {
		clearCommitTimeout();
		value = clampToRange(displayValue);
	}

	// Shared linear tick parameters used for rendering and snapping
	function computeLinearTickParams(): {
		niceTickSize: number;
		niceMin: number;
		majorTickGroupingFactor: number;
		majorSpacing: number;
	} {
		const range = max - min;
		const currentTickCount = Math.max(2, tickCount);
		const unroundedTickSize = currentTickCount > 1 ? range / (currentTickCount - 1) : range;
		const exponent = Math.floor(Math.log(unroundedTickSize) / Math.log(base));
		const fraction = unroundedTickSize / Math.pow(base, exponent);

		let localSteps = [1];
		if (base >= 2) localSteps.push(2);
		const halfB = base / 2;
		if (base % 2 === 0 && halfB > (localSteps.at(-1) ?? 0) && halfB < base) {
			localSteps.push(halfB);
		}
		if (base === 10 && !localSteps.includes(5) && 5 < base && 5 > (localSteps.at(-1) ?? 0)) {
			localSteps.push(5);
		}
		if (base > (localSteps.at(-1) ?? 0)) localSteps.push(base);
		localSteps = [...new Set(localSteps)].sort((a, b) => a - b);

		let niceFraction = localSteps[localSteps.length - 1];
		for (let i = localSteps.length - 2; i >= 0; i--) {
			const s_curr = localSteps[i];
			const s_next = localSteps[i + 1];
			if (fraction < Math.sqrt(s_curr * s_next)) niceFraction = s_curr;
			else break;
		}

		const niceTickSize = niceFraction * Math.pow(base, exponent);
		const niceMin = Math.floor(min / niceTickSize) * niceTickSize;
		let majorTickGroupingFactor = base === 10 ? 5 : base % 2 === 0 && base > 2 ? base / 2 : base;
		if (majorTickGroupingFactor <= 1 && base > 1) majorTickGroupingFactor = base;
		const majorSpacing = majorTickGroupingFactor * niceTickSize;

		return { niceTickSize, niceMin, majorTickGroupingFactor, majorSpacing };
	}

	// Compute nearest major tick value for current config
	function getNearestMajorValue(targetValue: number): number {
		if (useLog) {
			const actualLogBase = base > 1 ? base : Math.E;
			const currentMinForLog = Math.max(EPSILON, min);
			if (max <= 0 || currentMinForLog <= 0) return clampToRange(targetValue);
			const k = Math.round(customLog(Math.max(currentMinForLog, targetValue), actualLogBase));
			const snapped = Math.pow(actualLogBase, k);
			return clampToRange(snapped);
		}
		// Linear: reuse the same params as rendering, and anchor majors at 0 like the renderer
		const range = max - min;
		if (range <= 0) return clampToRange(targetValue);
		const { majorSpacing } = computeLinearTickParams();
		if (!(majorSpacing > EPSILON)) return clampToRange(targetValue);

		const candidates: number[] = [];
		const startN = Math.ceil((min - EPSILON) / majorSpacing);
		const endN = Math.floor((max + EPSILON) / majorSpacing);
		for (let n = startN; n <= endN; n++) {
			candidates.push(n * majorSpacing);
		}
		// Include endpoints which are treated as majors in rendering logic
		candidates.push(min, max);

		let best = clampToRange(targetValue);
		let bestDist = Number.POSITIVE_INFINITY;
		for (const c of candidates) {
			const cc = clampToRange(c);
			const d = Math.abs(cc - targetValue);
			if (d < bestDist) {
				bestDist = d;
				best = cc;
			}
		}
		return best;
	}

	function positionToValue(position: number, altSnap: boolean = false): number {
		const clampedPosition = Math.max(0, Math.min(100, position));
		const currentMinForLog = Math.max(EPSILON, min);

		let rawValue;
		if (useLog) {
			const actualLogBase = base > 1 ? base : Math.E;
			const minL = customLog(currentMinForLog, actualLogBase);
			const maxL = customLog(max, actualLogBase);

			if (max <= 0 || currentMinForLog <= 0 || maxL <= minL) {
				return currentMinForLog;
			}
			const valueL = minL + (clampedPosition / 100) * (maxL - minL);
			rawValue = Math.pow(actualLogBase, valueL);
		} else {
			const range = max - min;
			if (range <= 0) return min;
			rawValue = min + (clampedPosition / 100) * range;
		}

		if (altSnap) {
			return getNearestMajorValue(rawValue);
		}

		if (step > 0) {
			return Math.round(rawValue / step) * step;
		}
		return Math.round(rawValue * 100) / 100;
	}

	// Handler for input field changes
	function handleInputChange(e: Event) {
		const target = e.target as HTMLInputElement;
		// Keep inputValueString in sync with what user types
		inputValueString = target.value;

		const parsedNum = parseFloat(inputValueString);
		if (!isNaN(parsedNum)) {
			const clampedNum = clampToRange(parsedNum);
			displayValue = clampedNum;
			// Update bindable immediately for typed input
			value = clampedNum;
			// Update last good value
			lastValidNumericValue = clampedNum;
		}
	}

	// Handler for input field blur
	function handleInputBlur() {
		isInputFocused = false;
		const parsedNum = parseFloat(inputValueString);

		if (!isNaN(parsedNum)) {
			const clampedNum = clampToRange(parsedNum);
			displayValue = clampedNum;
			lastValidNumericValue = clampedNum;
			flushCommit();
		} else {
			displayValue = lastValidNumericValue;
			flushCommit();
		}
		inputValueString = getFormattedInputValue(displayValue, useLog, step);
	}

	// Handler for keydown events on the input
	function handleInputKeyDown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			// Trigger blur, which will run handleInputBlur
			(e.target as HTMLInputElement).blur();
		}
	}

	let inputWidth = $derived(
		`${6 + getFormattedInputValue(max, useLog, step, { shortExponential: false }).length * 8}px`
	);

	// Drag Handling
	onMount(() => {
		let pointerCaptured = false;
		let pointerId: number | null = null;

		function handlePointerMove(e: PointerEvent) {
			if (!isDragging || !trackRef) return;
			e.preventDefault(); // Prevent scrolling during drag
			const rect = trackRef.getBoundingClientRect();
			const position = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
			scheduleCommit(positionToValue(position, e.altKey));
		}

		function endDrag() {
			if (isDragging) {
				isDragging = false;
				document.body.style.userSelect = '';
				document.body.classList.remove('cursor-grabbing');
				if (pointerCaptured && thumbRef && pointerId !== null) {
					thumbRef.releasePointerCapture(pointerId);
				}
				pointerCaptured = false;
				pointerId = null;
				flushCommit();
			}
		}

		function onPointerUpOrCancel(_e: PointerEvent) {
			endDrag();
		}

		function handleThumbPointerDown(e: PointerEvent) {
			if (!thumbRef) return;
			e.preventDefault();
			e.stopPropagation();
			isDragging = true;
			document.body.style.userSelect = 'none';
			document.body.classList.add('cursor-grabbing');
			pointerId = e.pointerId;
			try {
				thumbRef.setPointerCapture(pointerId);
				pointerCaptured = true;
			} finally {
				pointerCaptured = false;
			}
		}

		function handleTrackPointerDown(e: PointerEvent) {
			if ((e.target as HTMLElement)?.closest('[role="slider"]')) {
				return;
			}
			if (!trackRef) return;
			e.preventDefault();
			const rect = trackRef.getBoundingClientRect();
			const position = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
			scheduleCommit(positionToValue(position, e.altKey));

			isDragging = true;
			pointerId = e.pointerId;
			pointerCaptured = false;
			document.body.style.userSelect = 'none';
			document.body.classList.add('cursor-grabbing');

			thumbRef?.focus();
		}

		const currentThumbRef = thumbRef;
		const currentTrackRef = trackRef;

		const handleVisibilityChange = () => {
			if (document.visibilityState === 'hidden') {
				endDrag();
			}
		};

		if (currentThumbRef) {
			currentThumbRef.addEventListener('pointerdown', handleThumbPointerDown);
		}
		if (currentTrackRef) {
			currentTrackRef.addEventListener('pointerdown', handleTrackPointerDown);
		}

		document.addEventListener('pointermove', handlePointerMove);
		document.addEventListener('pointerup', onPointerUpOrCancel);
		document.addEventListener('pointercancel', onPointerUpOrCancel);
		document.addEventListener('visibilitychange', handleVisibilityChange);
		window.addEventListener('blur', endDrag);

		return () => {
			if (currentThumbRef) {
				currentThumbRef.removeEventListener('pointerdown', handleThumbPointerDown);
			}
			if (currentTrackRef) {
				currentTrackRef.removeEventListener('pointerdown', handleTrackPointerDown);
			}
			document.removeEventListener('pointermove', handlePointerMove);
			document.removeEventListener('pointerup', onPointerUpOrCancel);
			document.removeEventListener('pointercancel', onPointerUpOrCancel);
			document.removeEventListener('visibilitychange', handleVisibilityChange);
			window.removeEventListener('blur', endDrag);

			if (isDragging) {
				document.body.style.userSelect = '';
				document.body.classList.remove('cursor-grabbing');
			}
			clearCommitTimeout();
		};
	});

	const ticks = $derived.by(() => {
		if (!showTicks) return [];
		const tickElements: { key: string; class: string; style: string }[] = [];
		const C_EPSILON = 0.0001;
		// Use the same epsilon as the slider position mapping so tick range matches value range
		const currentMinForLog = Math.max(EPSILON, min);
		const actualLogBase = useLog && base > 1 ? base : Math.E;

		if (useLog) {
			const minVal = currentMinForLog;
			const maxVal = max;
			if (maxVal <= minVal || maxVal <= 0 || minVal <= 0) return [];

			const minL = customLog(minVal, actualLogBase);
			const maxL = customLog(maxVal, actualLogBase);
			if (maxL <= minL) return [];

			const getPositionPercent = (v: number) => {
				const valL = customLog(Math.max(minVal, v), actualLogBase);
				return ((valL - minL) / (maxL - minL)) * 100;
			};

			const orderSpan = maxL - minL;
			// Show dense log minors up to ~3 decades; beyond that, switch to a uniform grid
			const useUniformGrid = orderSpan > 3 + EPSILON;

			let currentPower = Math.floor(minL);
			let tickVal = Math.pow(actualLogBase, currentPower);

			while (tickVal <= maxVal * (actualLogBase * 0.9) && currentPower <= maxL + 1) {
				if (tickVal >= minVal / (actualLogBase * 0.9)) {
					const position = getPositionPercent(tickVal);
					if (position >= 0 && position <= 100) {
						tickElements.push({
							key: `major-log-${actualLogBase}-${currentPower.toFixed(2)}`,
							class: 'absolute w-px h-2 bg-neutral-500 transform -translate-x-1/2',
							style: `left: ${position}%; bottom: -4px;`
						});
					}
				}

				const nextMajorTickVal = Math.pow(actualLogBase, currentPower + 1);
				if (
					!useUniformGrid &&
					orderSpan > 0.5 &&
					nextMajorTickVal <= maxVal * actualLogBase &&
					nextMajorTickVal > tickVal * 1.1
				) {
					if (actualLogBase === 10) {
						const subdivisors = orderSpan > 4 ? [2, 5] : [2, 3, 4, 5, 6, 7, 8, 9];
						for (const i of subdivisors) {
							const minorTickValue = i * tickVal;
							if (
								minorTickValue > tickVal &&
								minorTickValue < nextMajorTickVal &&
								minorTickValue <= maxVal
							) {
								const minorPosition = getPositionPercent(minorTickValue);
								if (minorPosition >= 0 && minorPosition <= 100) {
									tickElements.push({
										key: `minor-log10-${currentPower.toFixed(2)}-${i}`,
										class: 'absolute w-px h-1 bg-neutral-400 transform -translate-x-1/2',
										style: `left: ${minorPosition}%; bottom: -3px;`
									});
								}
							}
						}
					} else if (actualLogBase > 1.5) {
						const midLogValue = customLog(tickVal, actualLogBase) + 0.5;
						const minorTickValue = Math.pow(actualLogBase, midLogValue);
						if (
							minorTickValue > tickVal &&
							minorTickValue < nextMajorTickVal &&
							minorTickValue <= maxVal
						) {
							const minorPosition = getPositionPercent(minorTickValue);
							if (minorPosition >= 0 && minorPosition <= 100) {
								tickElements.push({
									key: `minor-log-mid-${currentPower.toFixed(2)}`,
									class: 'absolute w-px h-1 bg-neutral-400 transform -translate-x-1/2',
									style: `left: ${minorPosition}%; bottom: -3px;`
								});
							}
						}
					}
				}
				currentPower++;
				tickVal = Math.pow(actualLogBase, currentPower);
				if (tickVal <= C_EPSILON && currentPower > minL + 5) break;
			}

			const majorCount = tickElements.filter((t) => t.class.includes('h-2')).length;
			if ((useUniformGrid || majorCount < 2) && orderSpan > C_EPSILON) {
				const numDisplayTicks = Math.min(Math.max(2, tickCount), 10);
				for (let i = 0; i <= numDisplayTicks; i++) {
					const fraction = i / numDisplayTicks;
					const currentLValue = minL + fraction * (maxL - minL);
					const tv = Math.pow(actualLogBase, currentLValue);
					const position = fraction * 100;

					const isMajor =
						i === 0 ||
						i === numDisplayTicks ||
						(tv > minVal &&
							tv < maxVal &&
							Math.abs(customLog(tv, actualLogBase) - Math.round(customLog(tv, actualLogBase))) <
								0.05);

					const key = `log-small-range-tick-${i}`;
					if (
						!tickElements.find(
							(t) =>
								Math.abs(parseFloat(t.style.match(/left: ([\d.]+)%;/)?.[1] || '-1000') - position) <
								0.1
						)
					) {
						tickElements.push({
							key,
							class: `absolute w-px ${isMajor ? 'h-2 bg-neutral-500' : 'h-1 bg-neutral-400'} transform -translate-x-1/2`,
							style: `left: ${position}%; bottom: ${isMajor ? '-4px' : '-3px'};`
						});
					}
				}
			}
		} else {
			const range = max - min;
			if (range <= 0) return [];

			const { niceTickSize, niceMin, majorTickGroupingFactor } = computeLinearTickParams();

			// If the tick size is too small, we just show the ticks at the min and max
			if (niceTickSize <= C_EPSILON) {
				if (min >= 0 && min <= 100)
					tickElements.push({
						key: 'linear-fallback-min',
						class: 'absolute w-px h-2 bg-neutral-500 transform -translate-x-1/2',
						style: `left: 0%; bottom: -4px;`
					});
				if (max >= 0 && max <= 100 && Math.abs(max - min) > C_EPSILON)
					tickElements.push({
						key: 'linear-fallback-max',
						class: 'absolute w-px h-2 bg-neutral-500 transform -translate-x-1/2',
						style: `left: 100%; bottom: -4px;`
					});
				return tickElements;
			}

			for (let tickVal = niceMin; tickVal <= max + niceTickSize / 2; tickVal += niceTickSize) {
				if (tickVal >= min - niceTickSize / 10) {
					const position = ((tickVal - min) / range) * 100;
					if (position >= -1 && position <= 101) {
						const isMinTick = Math.abs(tickVal - niceMin) < niceTickSize * 0.01;
						const isMaxTickCluster = tickVal >= max - niceTickSize * 0.1;

						const isIntermediateMajor =
							niceTickSize > C_EPSILON &&
							Math.abs(
								tickVal / (majorTickGroupingFactor * niceTickSize) -
									Math.round(tickVal / (majorTickGroupingFactor * niceTickSize))
							) < C_EPSILON;

						const isMajorTick =
							isMinTick ||
							isMaxTickCluster ||
							(tickVal > niceMin + niceTickSize * 0.1 &&
								tickVal < max - niceTickSize * 0.1 &&
								isIntermediateMajor);

						tickElements.push({
							key: `linear-tick-${tickVal.toFixed(5)}`,
							class: `absolute w-px ${isMajorTick ? 'h-2 bg-neutral-500' : 'h-1 bg-neutral-400'} transform -translate-x-1/2`,
							style: `left: ${Math.max(0, Math.min(100, position))}%; bottom: ${isMajorTick ? '-4px' : '-3px'};`
						});
					}
				}
			}
		}
		return tickElements;
	});
</script>

<div class="w-full {className}">
	{#if label}
		<FormLabel forInputId={id} value={label} />
	{/if}
	<div class="flex w-full gap-x-3 mb-1" class:mt-1={label}>
		<div class="flex items-top gap-2 w-full">
			<div class="relative flex-1 flex flex-col gap-0.5 px-0.5">
				<div bind:this={trackRef} class="relative cursor-pointer py-1 px-0.5 touch-none">
					<!-- Track -->
					<div class="absolute h-0.5 w-full bg-neutral-200"></div>

					<!-- Filled Track -->
					<div class="absolute h-0.5 bg-neutral-400" style:width="{sliderPosition}%"></div>

					<!-- Ticks -->
					{#if showTicks && ticks.length > 0}
						<div class="absolute w-full h-full bottom-0 pointer-events-none -translate-y-px">
							<!-- pointer-events-none so ticks don't interfere with track click -->
							{#each ticks as tick (tick.key)}
								<div class={tick.class} style={tick.style}></div>
							{/each}
						</div>
					{/if}

					<!-- Thumb -->
					<div
						bind:this={thumbRef}
						class="absolute w-2 h-4 bg-neutral-800 border-l border-r border-white cursor-grab active:cursor-grabbing transform -translate-x-1/2 -translate-y-1.75 touch-none"
						style:left="{sliderPosition}%"
						role="slider"
						aria-valuemin={min}
						aria-valuemax={max}
						aria-valuenow={displayValue}
						tabindex="0"
						onkeydown={(e) => {
							let stepAmount = step > 0 ? step : useLog ? displayValue * 0.05 : (max - min) / 100;
							if (stepAmount === 0 && max > min) stepAmount = (max - min) / 100;
							if (stepAmount === 0) stepAmount = EPSILON;

							if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
								scheduleCommit(clampToRange(displayValue - stepAmount));
							} else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
								scheduleCommit(clampToRange(displayValue + stepAmount));
							} else if (e.key === 'Home') {
								scheduleCommit(min);
							} else if (e.key === 'End') {
								scheduleCommit(max);
							}
						}}
					></div>
				</div>
				<div
					class="flex justify-between text-xs text-neutral-600 pt-1 -mx-0.5 font-mono font-semibold select-none"
				>
					<div>
						{tickFormatter
							? tickFormatter(min)
							: useLog
								? formatExponential(min, EXPONENTIAL_PRECISION)
								: min.toLocaleString()}
					</div>
					<div>
						{tickFormatter
							? tickFormatter(max)
							: useLog
								? formatExponential(max, EXPONENTIAL_PRECISION)
								: max.toLocaleString()}
					</div>
				</div>
			</div>
		</div>

		<!-- Number Input -->
		<div class="flex gap-1.5 items-center">
			<div class="flex h-7">
				<input
					type="text"
					bind:value={inputValueString}
					oninput={handleInputChange}
					onfocus={() => (isInputFocused = true)}
					onblur={handleInputBlur}
					onkeydown={handleInputKeyDown}
					style:width={inputWidth}
					class="bg-transparent text-controls-numeric text-right font-mono border-panel-border-base focus:border-neutral-500 border focus:outline-none h-full px-0.5"
					{min}
					{max}
				/>
				<!-- Unit, like "%" or "ms", if provided -->
				{#if unit}
					<span
						class="inline-flex items-center border border-l-0 border-panel-border-base bg-neutral-50 px-1 text-neutral-500 font-mono text-controls-numeric h-full"
					>
						{unit}
					</span>
				{/if}
			</div>

			{#if onReset}
				<ResetValueButton {hasDefaultValue} {onReset} />
			{/if}
		</div>
	</div>
</div>

<style>
	.cursor-grabbing {
		cursor: grabbing !important;
	}
	[role='slider'] {
		z-index: 10;
		touch-action: none;
	}
	/* Prevent touch scrolling on slider elements */
	.slider-track,
	.slider-thumb {
		touch-action: none;
	}
</style>
