<script lang="ts">
	import { CheckboxIcon } from 'example-common';

	let {
		items,
		groupName,
		isOn,
		onToggle
	}: {
		items: string[];
		groupName: string;
		isOn: (name: string) => boolean;
		onToggle: (name: string) => void;
	} = $props();

	function labelForMetric(localGroupName: string, metricName: string): string {
		if (metricName.startsWith(localGroupName + '/')) {
			return metricName.slice(localGroupName.length + 1);
		}
		const parts = metricName.split('/');
		return parts.slice(1).join('/') || parts[0];
	}
</script>

<div class="flex flex-wrap gap-1 items-center">
	{#each items as name (name)}
		<button
			type="button"
			class="px-1 py-0.5 border text-sm cursor-pointer flex items-center gap-1 {isOn(name)
				? 'bg-neutral-100 border-neutral-300 text-neutral-600'
				: 'bg-white border-neutral-200 text-neutral-500'}"
			aria-pressed={isOn(name)}
			onclick={() => onToggle(name)}
		>
			<div class="opacity-80">
				<CheckboxIcon checked={isOn(name)} />
			</div>
			{labelForMetric(groupName, name)}
		</button>
	{/each}
</div>
