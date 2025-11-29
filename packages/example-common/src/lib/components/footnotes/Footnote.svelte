<script lang="ts">
	import type { Snippet } from 'svelte';

	import { getContext } from 'svelte';

	const FOOTNOTES_CTX = 'sequence-toy-footnotes';

	let { id, label, children }: { id?: string; label?: string | number; children: Snippet } =
		$props();

	let key = $state<string | null>(null);
	let computedLabel = $state<string | null>(null);

	$effect(() => {
		if (!key) {
			const ctx = getContext<
				| {
						register: (e: {
							key?: string;
							label?: string | number;
							content: Snippet;
						}) => { key: string; label: string };
				  }
				| undefined
			>(FOOTNOTES_CTX);
			if (ctx) {
				const res = ctx.register({ key: id, label, content: children });
				key = res.key;
				computedLabel = res.label;
			} else {
				computedLabel = label != null ? String(label) : '?';
			}
		}
	});
</script>

<sup class="ml-0.5 align-super text-[0.65em]">
	<!-- eslint-disable svelte/no-navigation-without-resolve -->
	<a
		href={key ? `#fn-${key}` : undefined}
		id={key ? `fnref-${key}` : undefined}
		role="doc-noteref"
		aria-describedby="footnote-label"
		class="no-underline">{computedLabel ?? (label != null ? String(label) : '?')}</a
	>
</sup>
