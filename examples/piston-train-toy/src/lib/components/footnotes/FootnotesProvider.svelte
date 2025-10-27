<script lang="ts">
	import type { Snippet } from 'svelte';

	import { setContext } from 'svelte';

	// Context key shared with Footnote.svelte
	const FOOTNOTES_CTX = 'sequence-toy-footnotes';

	let { children, start = 1 }: { children: Snippet; start?: number } = $props();

	type Entry = { key: string; label: string; content: Snippet };
	let entries = $state<Entry[]>([]);
	let nextIndex = $state(start);

	function register(entry: { key?: string; label?: string | number; content: Snippet }) {
		const resolvedKey = entry.key ?? `n${entries.length + 1}`;
		const resolvedLabel = entry.label != null ? String(entry.label) : String(nextIndex++);
		if (!entries.some((e) => e.key === resolvedKey)) {
			entries = [...entries, { key: resolvedKey, label: resolvedLabel, content: entry.content }];
		}
		return { key: resolvedKey, label: resolvedLabel };
	}

	setContext(FOOTNOTES_CTX, { register });
</script>

{@render children()}

<section class="mt-6" aria-labelledby="footnote-label" role="doc-endnotes">
	<h2 id="footnote-label" class="sr-only">Footnotes</h2>
	<ol class="text-sm text-neutral-600 space-y-1 list-none">
		{#each entries as e (e.key)}
			<li id={`fn-${e.key}`} role="doc-endnote" class="grid grid-cols-[auto_1fr] gap-x-3">
				<span class="text-neutral-500 tabular-nums">{e.label + '.'}</span>
				<div>
					{@render e.content()}
					<a href={`#fnref-${e.key}`} class="ml-1 no-underline" aria-label="Back to content">↩︎</a>
				</div>
			</li>
		{/each}
	</ol>
</section>
