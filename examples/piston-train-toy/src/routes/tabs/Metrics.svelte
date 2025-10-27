<script lang="ts">
	import MetricsSection from '$lib/components/metrics/MetricsSection.svelte';
	import RunChart from '$lib/components/metrics/RunChart.svelte';
	import ToggleChips from '$lib/components/ToggleChips.svelte';
	import { getMetricGroups, getMetricNamesFromLastNRuns } from '$lib/workspace/runs.svelte';
	import {
		metricsSectionsOpen,
		metricVisibility,
		toggleMetricsSection
	} from '$lib/workspace/ui.svelte';

	// Derived state: Automatically tracks metric names from the last 5 runs
	const metricNames = $derived(getMetricNamesFromLastNRuns(5));

	// Group metrics by prefix (everything before the first '/')
	const metricGroups = $derived(getMetricGroups(metricNames));

	function getDefaultMetricVisibility(name: string): boolean {
		if (name === 'speed/wall_clock_seconds') return false;
		return true;
	}

	function isMetricVisible(name: string): boolean {
		const userValue = metricVisibility.current[name];
		return userValue === undefined ? getDefaultMetricVisibility(name) : userValue;
	}

	function toggleMetricVisibility(name: string) {
		const current = isMetricVisible(name);
		metricVisibility.current = { ...metricVisibility.current, [name]: !current };
	}

	function getFilteredMetrics(groupName: string, metrics: string[]): string[] {
		return metrics.filter((m) => isMetricVisible(m));
	}
</script>

<div class="p-2.5 md:p-4 overflow-auto overscroll-contain flex flex-col flex-1 min-h-0">
	{#if Object.keys(metricGroups).length === 0}
		<div class="flex items-center justify-center h-32 text-neutral-500">
			<p>No metrics available. Start training to see charts.</p>
		</div>
	{:else}
		<div class="space-y-6">
			{#each Object.entries(metricGroups) as [groupName, metrics] (groupName)}
				{@const filteredMetrics = getFilteredMetrics(groupName, metrics)}
				{@const hasMetrics = filteredMetrics.length > 0}
				{@const sectionOpen = (metricsSectionsOpen.current[groupName] ?? true) && hasMetrics}
				<MetricsSection
					title={groupName}
					{groupName}
					isOpen={sectionOpen}
					enabled={hasMetrics}
					onToggle={hasMetrics ? () => toggleMetricsSection(groupName) : undefined}
				>
					{#snippet chips()}
						<ToggleChips
							items={metrics}
							{groupName}
							isOn={isMetricVisible}
							onToggle={toggleMetricVisibility}
						/>
					{/snippet}
					{#each filteredMetrics as metricName (metricName)}
						<div class="bg-white border border-neutral-200">
							<RunChart {metricName} />
						</div>
					{/each}
				</MetricsSection>
			{/each}
		</div>
	{/if}
</div>
