<script lang="ts">
	import type { ProfilingSnapshot } from '$lib/workspace/runs.svelte';
	import TensorChart from './TensorChart.svelte';
	import BufferChart from './BufferChart.svelte';

	interface Props {
		snapshot: ProfilingSnapshot | null;
	}

	let { snapshot }: Props = $props();

	// Debug: log when snapshot changes
	$effect(() => {
		if (snapshot) {
			console.log(
				`[ProfilingCharts] Received snapshot: step=${snapshot.step}, events=${snapshot.events.length}`
			);
			// Log category breakdown
			const categories = new Map<string, number>();
			for (const e of snapshot.events) {
				categories.set(e.category, (categories.get(e.category) || 0) + 1);
			}
			console.log('[ProfilingCharts] Event categories:', Object.fromEntries(categories));
		} else {
			console.log('[ProfilingCharts] Snapshot is null');
		}
	});

	// Shared state for cross-highlighting
	let highlightedBufferId = $state<string | null>(null);
	let highlightedTensorIds = $state<string[]>([]);

	// Shared dataZoom state for synchronized panning/zooming
	let dataZoomStart = $state(0);
	let dataZoomEnd = $state(100);

	// Calculate time range from events
	const timeRange = $derived.by(() => {
		if (!snapshot || snapshot.events.length === 0) {
			return { min: 0, max: 100, duration: 100 };
		}
		const times = snapshot.events.map((e) => e.start_us / 1000);
		const endTimes = snapshot.events.map((e) => (e.start_us + e.duration_us) / 1000);
		const minTime = Math.min(...times);
		const maxTime = Math.max(...endTimes);
		return { min: minTime, max: maxTime, duration: maxTime - minTime };
	});

	// Check if we have tensor events
	const hasTensorEvents = $derived.by(() => {
		if (!snapshot) return false;
		return snapshot.events.some(
			(e) => e.category === 'tensor_allocation' || e.category === 'tensor_deallocation'
		);
	});

	function handleTensorHover(bufferId: string | null, tensorIds: string[]) {
		highlightedBufferId = bufferId;
		highlightedTensorIds = tensorIds;
	}

	function handleBufferHover(bufferId: string | null, tensorIds: string[]) {
		highlightedBufferId = bufferId;
		highlightedTensorIds = tensorIds;
	}

	function handleDataZoomChange(start: number, end: number) {
		dataZoomStart = start;
		dataZoomEnd = end;
	}
</script>

<div class="w-full space-y-2">
	<!-- Header with step and time info -->
	<div class="flex items-center justify-between px-2">
		<div class="text-xs text-slate-500">
			{#if snapshot && timeRange}
				Step {snapshot.step} · {snapshot.events.length} events · {timeRange.min.toFixed(0)}–{timeRange.max.toFixed(
					0
				)} ms
			{:else if snapshot}
				Step {snapshot.step} · {snapshot.events.length} events
			{:else}
				No profiling data
			{/if}
		</div>
	</div>

	{#if snapshot && snapshot.events.length > 0}
		<!-- Tensor Chart (top) -->
		{#if hasTensorEvents}
			<TensorChart
				events={snapshot.events}
				minTime={timeRange.min}
				maxTime={timeRange.max}
				{highlightedBufferId}
				onTensorHover={handleTensorHover}
				onDataZoomChange={handleDataZoomChange}
				{dataZoomStart}
				{dataZoomEnd}
			/>
		{/if}

		<!-- Buffer Chart (bottom) -->
		<BufferChart
			events={snapshot.events}
			minTime={timeRange.min}
			maxTime={timeRange.max}
			{highlightedBufferId}
			{highlightedTensorIds}
			onBufferHover={handleBufferHover}
			onDataZoomChange={handleDataZoomChange}
			{dataZoomStart}
			{dataZoomEnd}
		/>
	{:else}
		<div class="text-center text-slate-400 py-8">No profiling data available</div>
	{/if}
</div>

