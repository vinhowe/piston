<script lang="ts">
	import { clearPastRuns, runsMap } from '$lib/workspace/runs.svelte';
	import { getIconStrokeWidth } from '$lib/workspace/ui.svelte';
	import { Trash } from '@lucide/svelte/icons';

	const runs = $derived(Array.from(runsMap.values()).toReversed());

	const frozenColumn = { key: 'runId', label: 'Name' };

	const iconStrokeWidth = $derived(getIconStrokeWidth());
</script>

<div>
	{#if runs.length > 1}
		<button
			type="button"
			class="text-neutral-800 cursor-pointer text-sm p-1 bg-neutral-100 w-full flex items-center gap-1"
			onclick={clearPastRuns}
		>
			<Trash class="inline-block h-3.5 w-3.5 -translate-y-[0.5px]" strokeWidth={iconStrokeWidth} />
			Clear past runs
		</button>
	{/if}

	<div class="overflow-x-clip">
		<div class="-m-px">
			<div class="max-h-64 overflow-auto">
				<table
					class="text-sm table-fixed border-spacing-0 w-max min-w-full [&_th]:border [&_th]:border-panel-border-base [&_th]:truncate [&_td]:border [&_td]:border-panel-border-base"
				>
					<colgroup>
						<col class="w-34" />
					</colgroup>
					<thead class="text-neutral-700 bg-neutral-100">
						<tr class="[&_th]:px-1 [&_th]:py-0.5 [&_th]:text-left [&_th]:font-medium">
							<th class="sticky top-0 z-30 truncate bg-neutral-100">{frozenColumn.label}</th>
							<th class="sticky top-0 z-20 bg-neutral-100">Changes</th>
						</tr>
					</thead>
					<tbody class="[&_td]:px-1 [&_td]:py-0.5">
						{#each runs as run (run.runId)}
							<tr>
								<td class="z-10 bg-white truncate" title={run.runId}>{run.runId}</td>
								<td class="whitespace-nowrap">{run.diffSummary ?? 'initial experiment'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
	</div>
</div>
