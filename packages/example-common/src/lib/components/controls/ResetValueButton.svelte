<script lang="ts">
	import { UndoIcon } from '@lucide/svelte/icons';

	let {
		hasDefaultValue = false,
		onReset
	}: {
		hasDefaultValue: boolean;
		onReset: () => void;
	} = $props();
</script>

<button
	class="w-3.75 h-3.75 text-neutral-600 disabled:text-neutral-300 enabled:cursor-pointer"
	title="Reset to default"
	disabled={hasDefaultValue}
	onclick={() => {
		onReset?.();
	}}
>
	<UndoIcon class="w-3.5 h-3.5" strokeWidth={2.75} />
</button>
