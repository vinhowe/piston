<script lang="ts">
	import RadioGroupInput from '$lib/components/controls/radio/RadioGroupInput.svelte';
	import MetricsSection from '$lib/components/metrics/MetricsSection.svelte';
	import RunChart from '$lib/components/metrics/RunChart.svelte';
	import ValidationCompletionsViewer from '$lib/components/metrics/validationCompletions/ValidationCompletionsViewer.svelte';
	import ToggleChips from '$lib/components/ToggleChips.svelte';
	import UserGuideTooltip from '$lib/components/UserGuideTooltip.svelte';
	import { config } from '$lib/workspace/config.svelte';
	import {
		getCurrentRun,
		getMetricGroups,
		getMetricNamesFromLastNRuns
	} from '$lib/workspace/runs.svelte';
	import {
		getIconStrokeWidth,
		metricsSectionsOpen,
		metricVisibility,
		toggleMetricsSection,
		tourState
	} from '$lib/workspace/ui.svelte';
	import { sortWithPriority } from '$lib/workspace/utils';
	import { updateVisualizerTarget } from '$lib/workspace/workers.svelte';
	import { BotMessageSquare, ExternalLink } from 'lucide-svelte';

	import Visualize from '../../lib/components/Visualize.svelte';

	const iconStrokeWidth = $derived(getIconStrokeWidth());

	// Derived state: Automatically tracks metric names from the last 5 runs
	const metricNames = $derived(getMetricNamesFromLastNRuns(5));

	// Group metrics by prefix (everything before the first '/')
	const metricGroups = $derived(getMetricGroups(metricNames));

	const runConfig = $derived(getCurrentRun()?.config);

	// Derived: lr scheduler present
	const lrSchedulerPresent = $derived(
		(runConfig?.optimizer?.lrScheduler?.present ?? false) ||
			(runConfig?.optimizer?.warmupSteps?.present ?? false)
	);

	function getDefaultMetricVisibility(name: string): boolean {
		if (name === 'allocation/active_tensor_count') return false;
		if (name === 'speed/wall_clock_seconds') return false;
		if (name === 'optimizer/learning_rate') return !!lrSchedulerPresent;
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

	function getSortedMetrics(groupName: string, metrics: string[]): string[] {
		let priorities: string[] = [];
		if (groupName === 'validation') {
			priorities = ['validation/completions', 'validation/accuracy'];
		} else if (groupName === 'allocation') {
			priorities = ['allocation/gpu_memory_mb'];
		} else if (groupName === 'train') {
			priorities = ['train/loss'];
		}
		return sortWithPriority(metrics, (m) => m, priorities);
	}

	function getFilteredMetrics(groupName: string, metrics: string[]): string[] {
		const sortedMetrics = getSortedMetrics(groupName, metrics);
		return sortedMetrics.filter((m) => isMetricVisible(m));
	}
</script>

<div class="p-2.5 md:p-4 overflow-auto overscroll-contain flex flex-col flex-1 min-h-0">
	{#if !tourState.current.restartedExperiment}
		<UserGuideTooltip icon={BotMessageSquare} class="mb-5 @md:mb-7">
			Tinker with the experiment setup, then click <span class="font-semibold">New Changes</span>
			to try out your changes. You can probably break it!
			<a
				href="https://github.com/vinhowe/piston/issues"
				class="underline"
				target="_blank"
				rel="noopener noreferrer"
				>Report issues on Github <ExternalLink
					class="inline-block h-3.5 w-3.5 -translate-y-0.5"
					strokeWidth={iconStrokeWidth}
				/></a
			>.
		</UserGuideTooltip>
	{/if}
	{#if Object.keys(metricGroups).length === 0}
		<div class="flex items-center justify-center h-32 text-neutral-500">
			<p>No metrics available. Start training to see charts.</p>
		</div>
	{:else}
		<div class="space-y-6">
			{#if config.training.enableVisualization}
				<MetricsSection
					title="Visualize"
					groupName="visualize"
					contentClass=""
					isOpen={metricsSectionsOpen.current['visualize'] ?? true}
					onToggle={() => toggleMetricsSection('visualize')}
				>
					{#snippet chips()}
						{#if (runConfig ?? config).training.validation.present}
							<RadioGroupInput
								id="visualize-target"
								name="visualize"
								class="self-end -translate-y-[0.5px]"
								bind:value={() => config.visualization.target, updateVisualizerTarget}
								options={[
									{ value: 'validation', label: 'Selected Validation Example' },
									{ value: 'train', label: 'Training Batch Example' }
								]}
							/>
						{/if}
					{/snippet}
					<Visualize />
				</MetricsSection>
			{/if}

			{#each Object.entries(metricGroups).sort(([a], [b]) => {
				const order = ['validation', 'train', 'optimizer'];
				const aPriority = order.indexOf(a);
				const bPriority = order.indexOf(b);
				return (aPriority === -1 ? 999 : aPriority) - (bPriority === -1 ? 999 : bPriority) || a.localeCompare(b);
			}) as [groupName, metrics] (groupName)}
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
							items={getSortedMetrics(groupName, metrics)}
							{groupName}
							isOn={isMetricVisible}
							onToggle={toggleMetricVisibility}
						/>
					{/snippet}
					{#each filteredMetrics as metricName (metricName)}
						<div
							class="bg-white border border-neutral-200 {metricName === 'validation/completions'
								? '@3xl:col-span-2'
								: ''}"
						>
							{#if metricName === 'validation/completions'}
								<ValidationCompletionsViewer />
							{:else}
								<RunChart {metricName} />
							{/if}
						</div>
					{/each}
				</MetricsSection>
			{/each}
		</div>
	{/if}
</div>
