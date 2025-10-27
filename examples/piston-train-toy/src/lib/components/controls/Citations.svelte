<script lang="ts">
	type CitationEntry = { name: string; url?: string };

	export type CitationEntries = {
		entries?: readonly CitationEntry[];
		extra?: string;
	};

	let { citations }: { citations: CitationEntries } = $props();
</script>

<span class="text-2xs text-neutral-500">
	{#each citations?.entries ?? [] as e, i (e.name)}
		{#if i > 0},
		{/if}
		{#if e.url}
			<!-- eslint-disable svelte/no-navigation-without-resolve -->
			<a
				href={e.url}
				target="_blank"
				rel="noopener noreferrer"
				class="underline underline-offset-2 decoration-dotted">{e.name}</a
			>
		{:else}
			{e.name}
		{/if}
	{/each}{#if citations?.extra}{citations.extra}{/if}
</span>
