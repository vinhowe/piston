<script lang="ts">
	import KatexBlock from '../KatexBlock.svelte';

	type $$Props = {
		forInputId?: string;
		class?: string;
		value: string;
	};
	let { forInputId, class: additionalClasses = '', value }: $$Props = $props();
</script>

<label
	for={forInputId}
	class={`block text-neutral-700 text-sm font-medium ${additionalClasses}`.trim()}
>
	<KatexBlock text={value} />
</label>
