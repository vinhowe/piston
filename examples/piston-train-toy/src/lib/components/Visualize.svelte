<script lang="ts">
	import type { MatchBox } from '$lib/train/visualizer';

	import KatexBlock from '$lib/components/KatexBlock.svelte';
	import { config, equalsConfigDefault, resetConfigToDefaults } from '$lib/workspace/config.svelte';
	import {
		findExampleIdMatchingScript,
		getVisualizationExampleById,
		getVisualizationExampleOptions
	} from '$lib/workspace/visualizationExamples';
	import {
		getModelIndex,
		getVisualizerLayout,
		getWorkerVersion,
		initializeVisualizerCanvas,
		resizeVisualizer,
		trainingState,
		updateVisualizerScript,
		workerReady
	} from '$lib/workspace/workers.svelte';
	import {
		acceptCompletion,
		type CompletionContext,
		moveCompletionSelection,
		startCompletion
	} from '@codemirror/autocomplete';
	import { defaultHighlightStyle, syntaxHighlighting } from '@codemirror/language';
	import { linter, lintGutter } from '@codemirror/lint';
	import { Compartment, EditorState, Prec, Transaction } from '@codemirror/state';
	import { Decoration, keymap, placeholder } from '@codemirror/view';
	import {
		completeCQL,
		cqlLanguage,
		cqlLanguageSupport,
		DiagnosticError,
		type LintDiagnostic,
		parse,
		type TensorQuery,
		validateScriptAgainstIndex
	} from '@piston-ml/piston-web';
	import { basicSetup, EditorView } from 'codemirror';
	import {
		Minimize2Icon,
		PlayIcon,
		RotateCcwIcon,
		SquarePenIcon,
		ZoomInIcon,
		ZoomOutIcon
	} from 'lucide-svelte';
	import { onMount } from 'svelte';

	let editorContainer: HTMLDivElement | null = $state(null);
	let editorView: EditorView | null = $state(null);
	let canvasEl: HTMLCanvasElement | null = $state(null);
	let canvasContainerEl: HTMLDivElement | null = $state(null);
	let offscreenInitializationState = $state<'idle' | 'initializing' | 'ready' | 'error'>('idle');
	let resizeObserver: ResizeObserver | null = $state(null);
	let overflowResizeObserver = $state<ResizeObserver | null>(null);
	let isMinimized = $state(true);
	let lastAppliedScript: string | null = $state(null);
	const effectiveScript = $derived(
		(config.visualization.example === 'custom'
			? config.visualization.script
			: getVisualizationExampleById(config.visualization.example).script) as string
	);
	const scriptChangedSinceApply = $derived(effectiveScript !== lastAppliedScript);
	let hoveredQueryIndex: number | null = $state(null);
	let selectedQueryIndex: number | null = $state(null);
	let selectedBoxIndex: number | null = $state(null);
	let minimizedOffsets: Array<{ from: number; to: number }> = $state([]);
	// Prevent selection feedback loops when we programmatically move the editor cursor
	let suppressSelectionSync = false;
	// Track last applied highlight to avoid redundant reconfigures and throttle updates
	let lastHighlight: { from: number; to: number; cls: string } | null = null;
	let highlightRaf = 0;
	let validatedQueries: TensorQuery[] | null = $state(null);
	// Debounce hover changes slightly to reduce thrash on rapid enter/leave
	let hoverDebounceTimer: ReturnType<typeof setTimeout> | null = null;
	function setHoveredQueryIndexDebounced(newIndex: number | null) {
		if (hoverDebounceTimer) clearTimeout(hoverDebounceTimer);
		hoverDebounceTimer = setTimeout(() => {
			hoveredQueryIndex = newIndex;
		}, 12);
	}

	function validateScript(doc: string): {
		valid: boolean;
		diagnostics: LintDiagnostic[];
	} {
		// Parse diagnostics
		let parseDiagnostics: LintDiagnostic[] = [];
		let queries: TensorQuery[] | null = null;
		try {
			queries = parse(doc);
		} catch (e) {
			validatedQueries = null;
			if (e instanceof DiagnosticError) {
				parseDiagnostics = [e.toLintDiagnostic()];
			} else {
				const message = e instanceof Error ? e.message : String(e);
				parseDiagnostics = [{ from: 0, to: Math.max(0, doc.length), message, severity: 'error' }];
			}
		}

		if (!queries) return { valid: false, diagnostics: parseDiagnostics };

		const indexDiagnostics = modelIndex ? validateScriptAgainstIndex(queries, modelIndex) : [];
		validatedQueries = queries;

		return {
			valid: parseDiagnostics.length === 0,
			diagnostics: [...parseDiagnostics, ...indexDiagnostics]
		};
	}

	// Inline CQL linter: parses the document and returns diagnostics; also updates validation status.
	const cqlLinter = linter((view) => {
		const doc = view.state.doc.toString();
		const { diagnostics } = validateScript(doc);
		return diagnostics.map((d) => ({
			from: d.from,
			to: d.to,
			message: d.message,
			severity: d.severity,
			source: d.source ?? 'cql'
		}));
	});

	// Compartment to allow dynamic reconfiguration between minimized and full editor modes
	const modeCompartment = new Compartment();
	const highlightCompartment = new Compartment();
	const smallerTextTheme = EditorView.theme({
		'&.cm-editor': {
			fontSize: '12px',
			fontFamily: 'Berkeley Mono Variable'
		}
	});

	function measureAndResize() {
		const parent = canvasEl?.parentElement as HTMLElement | null;
		const rect = (parent ?? canvasEl)?.getBoundingClientRect();
		if (!rect) return;
		const w = Math.max(1, rect.width | 0);
		resizeVisualizer(w);
	}

	function setupResizeObserver() {
		if (!canvasEl) return;
		const parent = canvasEl.parentElement as HTMLElement | null;
		if (!parent) return;
		try {
			resizeObserver?.disconnect();
		} catch (e) {
			void e;
		}
		resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const { width } = entry.contentRect;
				const w = Math.max(1, width | 0);
				resizeVisualizer(w);
			}
		});
		resizeObserver.observe(parent);
		measureAndResize();
	}

	function replaceEditorDocument(newDoc: string) {
		if (!editorView) return;
		const current = editorView.state.doc.toString();
		if (current === newDoc) return;
		editorView.dispatch({
			changes: { from: 0, to: editorView.state.doc.length, insert: newDoc }
		});
	}

	function computeMinimizedDocAndOffsets(sourceDoc: string): {
		doc: string;
		offsets: Array<{ from: number; to: number }>;
	} {
		let list: TensorQuery[];
		try {
			list = parse(sourceDoc);
		} catch (_e) {
			return { doc: '"Couldn\'t parse script; edit to fix"', offsets: [] };
		}
		if (!list || list.length === 0) return { doc: '', offsets: [] };
		const parts: string[] = [];
		const offsets: Array<{ from: number; to: number }> = [];
		let offset = 0;
		for (let i = 0; i < list.length; i++) {
			const b = list[i];
			const lines: string[] = [];
			if (b.label) {
				lines.push(`// ${b.label}`);
			}
			const statement = b.parsedQuery.source
				.split(/\r?\n/)
				.map((line: string) => line.trim())
				.join(' ');
			lines.push(statement);
			const block = lines.join('\n');
			const from = offset;
			const to = from + block.length;
			offsets[i] = { from, to };
			parts.push(block);
			offset = to + (i < list.length - 1 ? 1 : 0);
		}
		return { doc: parts.join('\n'), offsets };
	}

	let suppressDocSync = false;

	function buildFullExtensions() {
		return [
			Prec.highest(
				keymap.of([
					{ key: 'Ctrl-Space', run: startCompletion, preventDefault: true },
					{ key: 'Tab', run: acceptCompletion, preventDefault: true },
					{ key: 'Ctrl-n', run: moveCompletionSelection(true) },
					{ key: 'Ctrl-p', run: moveCompletionSelection(false) },
					{ key: 'ArrowDown', run: moveCompletionSelection(true) },
					{ key: 'ArrowUp', run: moveCompletionSelection(false) },
					{ key: 'PageDown', run: moveCompletionSelection(true, 'page') },
					{ key: 'PageUp', run: moveCompletionSelection(false, 'page') }
				])
			),
			basicSetup,
			cqlLinter,
			lintGutter(),
			placeholder('Press Ctrl-Space to autocomplete'),
			EditorView.updateListener.of((update) => {
				if (update.docChanged) {
					const newDoc = update.state.doc.toString();
					if (suppressDocSync) return;
					const matchId = findExampleIdMatchingScript(newDoc);
					if (matchId) {
						if (config.visualization.example !== matchId) {
							config.visualization.example = matchId;
						}
					} else {
						if (config.visualization.example !== 'custom') {
							config.visualization.example = 'custom';
						}
						if (config.visualization.script !== newDoc) {
							config.visualization.script = newDoc;
						}
					}
				}
			}),
			EditorView.updateListener.of((update) => {
				if (
					update.selectionSet &&
					!isMinimized &&
					!scriptChangedSinceApply &&
					!suppressSelectionSync
				) {
					const hasUserEvent = update.transactions.some(
						(tr) => !!tr.annotation(Transaction.userEvent)
					);
					if (!hasUserEvent) return;
					const pos = update.state.selection.main.head;
					let idx: number | null = null;
					for (let i = 0; i < queries.length; i++) {
						const q = queries[i];
						if (pos >= (q?.parsedQuery.from ?? 0) && pos < (q?.parsedQuery.to ?? 0)) {
							idx = i;
							break;
						}
					}
					if (selectedQueryIndex !== idx) {
						selectedQueryIndex = idx;
					}
				}
			}),
			smallerTextTheme
		];
	}

	function buildMinimizedExtensions() {
		return [
			EditorView.editable.of(false),
			EditorState.readOnly.of(true),
			// lineNumbers(),
			// foldGutter(),
			syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
			EditorView.theme({
				'&.cm-editor': { height: 'auto' }
				// '.cm-scroller': { overflow: 'hidden' }
			}),
			smallerTextTheme
		];
	}

	function reconfigureEditorMode() {
		if (!editorView || !smallerTextTheme) return;
		const extensions = isMinimized ? buildMinimizedExtensions() : buildFullExtensions();
		editorView.dispatch({ effects: modeCompartment.reconfigure(extensions) });
	}

	function expandToFull() {
		isMinimized = false;
		reconfigureEditorMode();
		replaceEditorDocument($state.snapshot(effectiveScript));
	}

	function minimizeToSummary() {
		isMinimized = true;
		reconfigureEditorMode();
		const r = computeMinimizedDocAndOffsets($state.snapshot(effectiveScript));
		minimizedOffsets = r.offsets;
		replaceEditorDocument(r.doc);
	}

	function applyEffectiveScript(scriptOverride?: string) {
		if (trainingState.current === 'stopped') return;
		const script = scriptOverride ?? $state.snapshot(effectiveScript);
		// Validate before sending to worker
		const { valid } = validateScript(script);
		if (!valid) return;
		updateVisualizerScript(config.visualization.example, script);
		lastAppliedScript = script;
	}

	function selectVisualizationExample(id: string) {
		if (id === 'custom') {
			const currentDoc = editorView
				? editorView.state.doc.toString()
				: $state.snapshot(effectiveScript);
			config.visualization.example = 'custom';
			config.visualization.script = currentDoc;
		} else {
			config.visualization.example = id;
			config.visualization.script = null;
		}
		applyEffectiveScript();
	}

	const exampleOptions = $derived(getVisualizationExampleOptions(config));

	function moveEditorToQueryStart(idx: number | null | undefined) {
		if (idx == null || idx < 0) return;
		if (!editorView) return;
		const range = isMinimized ? minimizedOffsets?.[idx] : queries?.[idx].parsedQuery;
		if (!range) return;
		// Avoid feedback loop: mark this as a programmatic selection
		suppressSelectionSync = true;
		if (selectedQueryIndex !== idx) {
			selectedQueryIndex = idx;
		}
		editorView.dispatch({
			selection: { anchor: Math.max(0, range.from | 0) },
			scrollIntoView: true
		});
		// Release suppression on next frame after the editor applies the selection
		requestAnimationFrame(() => {
			suppressSelectionSync = false;
		});
	}

	const isMobileWidth = $derived(window.innerWidth < 768);
	const visualizerLayout = $derived(getVisualizerLayout());
	const canvasContainerHeight = $derived.by(() => {
		const h = visualizerLayout.height;
		const maxAutoHeight = isMobileWidth ? 200 : 300;
		if (typeof h === 'number' && h > 0) {
			return Math.min(maxAutoHeight, h + 2);
		}
		return maxAutoHeight;
	});
	let showBottomShadow = $state(false);

	function updateOverflowShadow() {
		const el = canvasContainerEl;
		if (!el) return;
		const tolerance = 1;
		const hasOverflow = el.scrollHeight > el.clientHeight + 0.5;
		const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - tolerance;
		showBottomShadow = hasOverflow && !atBottom;
	}

	function setupOverflowObserver() {
		if (!canvasContainerEl) return;
		try {
			overflowResizeObserver?.disconnect();
		} catch (e) {
			void e;
		}
		overflowResizeObserver = new ResizeObserver(() => updateOverflowShadow());
		overflowResizeObserver.observe(canvasContainerEl);
		updateOverflowShadow();
	}

	function formatScaleNumber(n: number): string {
		const rounded = Math.round(n * 1000) / 1000;
		return String(rounded)
			.replace(/\.0+$/, '')
			.replace(/(\.\d*[1-9])0+$/, '$1');
	}

	function adjustScale(factor: number) {
		if (!validatedQueries) return;
		const original = $state.snapshot(effectiveScript) ?? '';
		const edits: Array<{ from: number; to: number; text: string }> = [];
		for (const tq of validatedQueries) {
			const parsedQuery = tq.parsedQuery;
			const facets = parsedQuery.facets;
			const scale = facets?.scale;
			if (scale) {
				const isPercent = scale.unit === 'percent';
				const next = (scale.value ?? 0) * factor * (isPercent ? 100 : 1);
				const text = isPercent
					? `:scale(${formatScaleNumber(next)}%)`
					: `:scale(${formatScaleNumber(next)})`;
				edits.push({ from: scale.from, to: scale.to, text });
				continue;
			}

			const insertionText = ` :scale(${formatScaleNumber(factor)})`;
			if (facets) {
				edits.push({ from: facets.to, to: facets.to, text: insertionText });
				continue;
			}

			const sliceFrom = parsedQuery.slice?.from;
			const jsPipeFrom = parsedQuery.jsPipe?.from;
			let insertAt = parsedQuery.to;
			if (sliceFrom != null) insertAt = sliceFrom;
			else if (jsPipeFrom != null) insertAt = jsPipeFrom;
			edits.push({ from: insertAt, to: insertAt, text: insertionText });
		}

		if (edits.length === 0) return;
		edits.sort((a, b) => b.from - a.from);
		let result = original;
		for (const e of edits) {
			result = result.slice(0, e.from) + e.text + result.slice(e.to);
		}
		config.visualization.example = 'custom';
		config.visualization.script = result;
		requestAnimationFrame(() => applyEffectiveScript());
		if (!isMinimized) replaceEditorDocument(result);
	}

	const queries = $derived(visualizerLayout.queries ?? []);
	let __boxesKey: string | null = null;
	let __boxesMemo: (MatchBox & { fullyQualifiedPath: string })[] | null = null;
	const boxes = $derived.by(() => {
		const raw = visualizerLayout.boxes ?? null;
		if (!raw) {
			__boxesKey = null;
			__boxesMemo = null;
			return null;
		}
		const key = JSON.stringify(raw);
		if (key === __boxesKey && __boxesMemo) {
			return __boxesMemo;
		}
		const next = raw.map((box) => {
			const { match } = box;
			let fullyQualifiedPath = match.path;
			if (match.moduleSite) {
				fullyQualifiedPath = `${match.path} :${match.moduleSite}`;
			}
			if (match.op) {
				fullyQualifiedPath = `${fullyQualifiedPath} @ ${match.op}`;
			} else if (match.parameter) {
				fullyQualifiedPath = `${fullyQualifiedPath} #${match.parameter}`;
			}
			if (match.source.label) {
				fullyQualifiedPath = `${match.source.label} [${fullyQualifiedPath}]`;
			}
			if (match.source.scale && match.source.scale !== 1) {
				fullyQualifiedPath = `${fullyQualifiedPath} (${Math.round(match.source.scale * 10) / 10}x)`;
			}
			return { ...box, fullyQualifiedPath };
		}) as (MatchBox & { fullyQualifiedPath: string })[];
		__boxesKey = key;
		__boxesMemo = next;
		return next;
	});

	const labelHeight = 14;

	function avoidRightOverflow(box: MatchBox, isSelected: boolean) {
		return (node: HTMLElement) => {
			let raf = 0;
			const group = node.parentElement as HTMLElement | null;

			function measure() {
				raf = 0;
				// Reset before measuring so we get the natural right edge
				node.style.transform = '';
				node.style.maxWidth = '';
				const boundaryEl = node.closest('.relative') as HTMLElement | null;
				const boundaryRight = boundaryEl?.getBoundingClientRect().right ?? window.innerWidth;
				const rect = node.getBoundingClientRect();
				const overflow = rect.right - boundaryRight;
				if (overflow > 0) {
					node.style.transform = `translateX(-${Math.ceil(overflow + 2)}px)`;
				}
			}

			function onEnter() {
				if (raf) cancelAnimationFrame(raf);
				raf = requestAnimationFrame(measure);
			}

			function onLeave() {
				node.style.transform = '';
				node.style.maxWidth = `${box.width}px`;
			}

			function onResize() {
				if ((group && group.matches(':hover')) || isSelected) {
					if (raf) cancelAnimationFrame(raf);
					raf = requestAnimationFrame(measure);
				}
			}

			// If currently selected, ensure we measure immediately to avoid overflow.
			if (isSelected) {
				if (raf) cancelAnimationFrame(raf);
				raf = requestAnimationFrame(measure);
			} else {
				if (!(group && group.matches(':hover'))) {
					onLeave();
				}
				group?.addEventListener('mouseenter', onEnter);
				group?.addEventListener('mouseleave', onLeave);
				window.addEventListener('resize', onResize);
			}

			return () => {
				if (!isSelected) {
					group?.removeEventListener('mouseenter', onEnter);
					group?.removeEventListener('mouseleave', onLeave);
					window.removeEventListener('resize', onResize);
				}
				if (raf) cancelAnimationFrame(raf);
			};
		};
	}

	$effect(() => {
		if (canvasEl && workerReady.current && offscreenInitializationState === 'idle') {
			offscreenInitializationState = 'initializing';
			try {
				initializeVisualizerCanvas(canvasEl, labelHeight);
				// initial size
				const rect = canvasEl.getBoundingClientRect();
				const w = Math.max(1, rect.width | 0);
				resizeVisualizer(w);
				setupResizeObserver();
				offscreenInitializationState = 'ready';
			} catch (e) {
				console.error('Failed to init visualizer canvas', e);
				offscreenInitializationState = 'error';
			}
		}
	});

	// Recreate the canvas and force re-init when the worker restarts
	const workerVersion = $derived.by(() => getWorkerVersion().current);
	$effect(() => {
		void workerVersion;
		offscreenInitializationState = 'idle';
		// Canvas DOM node is recreated under the keyed block; re-attach observer
		setTimeout(() => {
			setupResizeObserver();
			requestAnimationFrame(() => updateOverflowShadow());
		}, 0);
	});

	const modelIndex = $derived(getModelIndex());

	const cqlCompletions = cqlLanguage.data.of({
		autocomplete: (context: CompletionContext) =>
			modelIndex ? completeCQL(context, modelIndex) : null
	});

	onMount(() => {
		// Ensure the container exists before creating the editor
		if (!editorContainer) {
			console.error('Editor container element not found!');
			return;
		}

		// Initialize last applied script to current effective script on mount
		lastAppliedScript = $state.snapshot(effectiveScript);

		validateScript(effectiveScript);

		const initialDoc = isMinimized
			? (() => {
					const r = computeMinimizedDocAndOffsets($state.snapshot(effectiveScript));
					minimizedOffsets = r.offsets;
					return r.doc;
				})()
			: $state.snapshot(effectiveScript);
		const startState = EditorState.create({
			doc: initialDoc,
			extensions: [
				cqlLanguageSupport(cqlCompletions),
				modeCompartment.of(isMinimized ? buildMinimizedExtensions() : buildFullExtensions()),
				highlightCompartment.of(EditorView.decorations.of(Decoration.none))
			]
		});

		editorView = new EditorView({
			state: startState,
			parent: editorContainer
		});

		// Cleanup function to destroy the editor instance when the component unmounts
		return () => {
			editorView?.destroy();
		};
	});

	onMount(() => {
		setupOverflowObserver();
		window.addEventListener('resize', updateOverflowShadow);
		return () => {
			overflowResizeObserver?.disconnect();
			window.removeEventListener('resize', updateOverflowShadow);
		};
	});

	$effect(() => {
		if (!editorView) return;
		reconfigureEditorMode();
		if (isMinimized) {
			const r = computeMinimizedDocAndOffsets($state.snapshot(effectiveScript));
			minimizedOffsets = r.offsets;
			replaceEditorDocument(r.doc);
		} else {
			replaceEditorDocument($state.snapshot(effectiveScript));
		}
	});

	$effect(() => {
		void canvasContainerHeight;
		requestAnimationFrame(() => updateOverflowShadow());
	});

	$effect(() => {
		// Recalculate overlay visibility when the canvas content size changes
		void visualizerLayout.height;
		requestAnimationFrame(() => updateOverflowShadow());
	});

	$effect(() => {
		if (!editorView) return;
		const activeIndex = hoveredQueryIndex ?? selectedQueryIndex;

		// Compute desired highlight target
		let next: { from: number; to: number; cls: string } | null = null;
		if (!scriptChangedSinceApply && activeIndex != null) {
			const q = isMinimized ? minimizedOffsets?.[activeIndex] : queries?.[activeIndex]?.parsedQuery;
			if (q) {
				const from = Math.max(0, q.from | 0);
				const to = Math.max(from, q.to | 0);
				const baseQuery = queries?.[activeIndex];
				let cls = 'cm-query-highlight';
				if (baseQuery) {
					if (baseQuery.gradient) cls = 'cm-query-highlight-fuchsia';
					else if (baseQuery.target?.kind === 'parameter') cls = 'cm-query-highlight-teal';
					else cls = 'cm-query-highlight-amber';
				}
				next = { from, to, cls };
			}
		}

		// If no highlight desired
		if (!next) {
			if (lastHighlight !== null) {
				lastHighlight = null;
				editorView.dispatch({
					effects: highlightCompartment.reconfigure(EditorView.decorations.of(Decoration.none))
				});
			}
			return;
		}

		// If highlight unchanged, skip reconfigure
		if (
			lastHighlight &&
			lastHighlight.from === next.from &&
			lastHighlight.to === next.to &&
			lastHighlight.cls === next.cls
		) {
			return;
		}

		lastHighlight = next;
		// Throttle highlight updates with RAF to coalesce rapid hover in/out
		if (highlightRaf) cancelAnimationFrame(highlightRaf);
		highlightRaf = requestAnimationFrame(() => {
			const deco = Decoration.mark({ class: next!.cls }).range(next!.from, next!.to);
			const set = Decoration.set([deco], true);
			editorView?.dispatch({
				effects: highlightCompartment.reconfigure(EditorView.decorations.of(set))
			});
			highlightRaf = 0;
		});
	});
</script>

{#snippet boxLabel(
	box: MatchBox & { fullyQualifiedPath: string },
	type: 'activation' | 'parameter' | 'gradient',
	isHighlighted: boolean,
	isSelected: boolean
)}
	<div
		class="h-full ring-1 {isSelected
			? type === 'gradient'
				? 'bg-fuchsia-300/95 ring-fuchsia-300/95 text-black'
				: type === 'parameter'
					? 'bg-teal-300/95 ring-teal-300/95 text-black'
					: 'bg-amber-300/95 ring-amber-300/95 text-black'
			: isHighlighted
				? type === 'gradient'
					? 'ring-fuchsia-300/95 group-hover:bg-fuchsia-300/95 text-fuchsia-400'
					: type === 'parameter'
						? 'ring-teal-300/95 group-hover:bg-teal-300/95 text-teal-400'
						: 'ring-amber-300/95 group-hover:bg-amber-300/95 text-amber-400'
				: type === 'gradient'
					? 'ring-fuchsia-900/80 group-hover:bg-fuchsia-300/95 group-hover:ring-fuchsia-300/95 text-fuchsia-400'
					: type === 'parameter'
						? 'ring-teal-900/80 group-hover:bg-teal-300/95 group-hover:ring-teal-300/95 text-teal-400'
						: 'ring-amber-900/80 group-hover:bg-amber-300/95 group-hover:ring-amber-300/95 text-amber-400'} {isSelected
			? 'max-w-none w-max'
			: 'group-hover:max-w-none group-hover:w-max group-hover:text-black'} text-2xs md:text-xs px-0.5 font-mono leading-none"
	>
		<div class="overflow-clip flex items-baseline gap-1.25">
			<div class="h-full min-h-0" class:-translate-y-px={type === 'activation'}>
				<span class="text-[9px]">
					<KatexBlock
						text={type === 'gradient'
							? '$\\boldsymbol\\nabla$'
							: type === 'activation'
								? '$\\boldsymbol\\phi$'
								: '$\\boldsymbol\\theta$'}
					/>
				</span>
			</div>
			<span
				class="truncate group-hover:whitespace-nowrap {isSelected
					? 'font-bold'
					: 'group-hover:font-bold'}"
			>
				<span class={isSelected ? 'inline' : 'group-hover:inline hidden'}>
					({#if type === 'gradient'}
						gradient
					{:else if type === 'activation'}
						activation
					{:else}
						parameter
					{/if}):
				</span>
				<KatexBlock text={box.fullyQualifiedPath} />
			</span>
		</div>
	</div>
{/snippet}

<div class="flex flex-col gap-2">
	<div class="flex flex-col">
		<div
			class="flex flex-col md:flex-row items-start md:items-center bg-neutral-100 border border-panel-border-base -mb-px"
		>
			<div class="flex items-center gap-1 shrink-0 w-max">
				{#if trainingState.current !== 'stopped'}
					<button
						class="text-neutral-400 enabled:text-green-900 enabled:bg-gradient-to-t enabled:from-green-300 enabled:to-green-100 border enabled:border-green-600 border-transparent z-10 -my-px -ml-px p-1.5 cursor-pointer"
						disabled={!scriptChangedSinceApply}
						title="Apply script to current run"
						onclick={() => applyEffectiveScript()}
					>
						<PlayIcon class="w-3.5 h-3.5 shrink-0" />
					</button>
				{/if}
				<button
					class="py-1.5 px-1.75 text-neutral-700 disabled:text-neutral-400 enabled:cursor-pointer flex items-center select-none border-transparent"
					title="Reset to default"
					disabled={equalsConfigDefault('visualization.script') &&
						equalsConfigDefault('visualization.example')}
					onclick={() => {
						resetConfigToDefaults(['visualization.script', 'visualization.example']);
						requestAnimationFrame(() => applyEffectiveScript());
					}}
				>
					<RotateCcwIcon class="w-3.5 h-3.5 shrink-0" />
				</button>
				{#if !isMinimized}
					<button
						class="text-neutral-700 p-1.5 cursor-pointer flex items-center select-none border-transparent"
						title="Minimize"
						onclick={() => minimizeToSummary()}
					>
						<Minimize2Icon class="w-3.5 h-3.5 shrink-0" />
					</button>
				{/if}
				{#if isMinimized}
					<div
						class="text-purple-700 cursor-pointer flex items-center select-none text-xs font-medium tracking-wide px-1.5 h-[calc(100%+2px)] -my-px z-10 gap-1.5 shrink-0"
						title="Edit script"
						onclick={() => expandToFull()}
						tabindex="0"
						role="button"
						aria-label="Edit script"
						onkeydown={(e) => {
							if (e.key === 'Enter' || e.key === ' ') {
								e.preventDefault();
								expandToFull();
							}
						}}
					>
						<SquarePenIcon class="w-3.5 h-3.5 shrink-0" />
						Click script to edit
					</div>
				{/if}
			</div>
			<div
				class="flex items-center gap-1 px-0.75 md:px-1 py-0.75 md:py-0.5 -mt-0.75 md:mt-0 overflow-x-auto flex-nowrap [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden w-full md:w-max"
			>
				{#each exampleOptions as opt (opt.value)}
					{@const selected = opt.value === config.visualization.example}
					<button
						class="text-xs px-1 border cursor-pointer leading-none py-0.5 tracking-wide whitespace-nowrap"
						id={`visualization-example-${opt.value}`}
						class:font-medium={selected}
						class:bg-green-200={selected}
						class:text-neutral-900={selected}
						class:border-green-500={selected}
						class:bg-green-50={!selected}
						class:text-green-700={!selected}
						class:border-green-300={!selected}
						title={String(opt.text)}
						onclick={() => selectVisualizationExample(String(opt.value))}
					>
						{String(opt.text)}
					</button>
				{/each}
			</div>
		</div>
		<div
			bind:this={editorContainer}
			class={isMinimized
				? 'w-full border border-neutral-300 overflow-hidden cursor-pointer'
				: 'h-[40vh] min-h-[10rem] w-full border border-neutral-300 resize-y overflow-auto'}
			onclick={() => {
				if (isMinimized) expandToFull();
			}}
			role="button"
			tabindex="0"
			aria-label="Expand editor"
			onkeydown={(e) => {
				if (!isMinimized) return;
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					expandToFull();
				}
			}}
		></div>
	</div>
	<div class="relative">
		<div
			bind:this={canvasContainerEl}
			class="relative w-full border border-neutral-300 resize-y overflow-auto bg-black"
			onscroll={() => updateOverflowShadow()}
			style:height={`${canvasContainerHeight}px`}
			onclick={() => {
				selectedQueryIndex = null;
				selectedBoxIndex = null;
			}}
			onkeydown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					selectedQueryIndex = null;
					selectedBoxIndex = null;
				}
			}}
			role="button"
			tabindex="0"
			aria-label="Clear selection"
		>
			{#key workerVersion}
				<canvas
					bind:this={canvasEl}
					class="block"
					style="image-rendering: pixelated; width: {visualizerLayout.width
						? `${visualizerLayout.width}px`
						: '100%'}; height: {visualizerLayout.height ? `${visualizerLayout.height}px` : 'auto'};"
				></canvas>
				{#if boxes}
					{#each boxes as box, i (i)}
						{@const isHighlighted =
							!scriptChangedSinceApply &&
							(hoveredQueryIndex === (box.match.queryIndex ?? -1) ||
								selectedQueryIndex === (box.match.queryIndex ?? -1))}
						{@const isSelected = selectedBoxIndex === i}
						<div
							class="group cursor-pointer"
							onmouseenter={() => {
								if (!scriptChangedSinceApply)
									setHoveredQueryIndexDebounced(box.match.queryIndex ?? null);
							}}
							onmouseleave={() => {
								if (!scriptChangedSinceApply) setHoveredQueryIndexDebounced(null);
							}}
							onclick={(e) => {
								e.stopPropagation();
								moveEditorToQueryStart(box.match.queryIndex ?? null);
								selectedBoxIndex = i;
							}}
							role="button"
							tabindex="0"
							onkeydown={(e) => {
								e.stopPropagation();
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									moveEditorToQueryStart(box.match.queryIndex ?? null);
									selectedBoxIndex = i;
								}
							}}
						>
							<div
								{@attach avoidRightOverflow(box, isSelected)}
								class="absolute {isSelected ? 'z-20' : 'group-hover:z-10'}"
								style={`left:${box.x}px;top:${box.y - labelHeight}px;max-width:${box.width}px;height:${labelHeight - 1}px`}
								aria-hidden={true}
							>
								{@render boxLabel(
									box,
									box.match.source.gradient
										? 'gradient'
										: box.match.type === 'parameter'
											? 'parameter'
											: 'activation',
									isHighlighted,
									isSelected
								)}
							</div>
							<!-- Invisible bridge to make hover/click target continuous -->
							<div
								class="absolute"
								style={`left:${box.x}px;top:${box.y - 1}px;width:${box.width}px;height:2px`}
								aria-hidden={true}
							></div>
							<div
								class="absolute ring-1 {isHighlighted || isSelected
									? box.match.source.gradient
										? 'ring-fuchsia-300/95'
										: box.match.type === 'parameter'
											? 'ring-teal-300/95'
											: 'ring-amber-300/95'
									: box.match.source.gradient
										? 'ring-fuchsia-900/80 group-hover:ring-fuchsia-300/95'
										: box.match.type === 'parameter'
											? 'ring-teal-900/80 group-hover:ring-teal-300/95'
											: 'ring-amber-900/80 group-hover:ring-amber-300/95'}"
								style={`left:${box.x}px;top:${box.y}px;width:${box.width}px;height:${box.height}px`}
								title={`${box.match.path} [${box.match.shape.join('x')}]`}
							></div>
						</div>
					{/each}
				{/if}
			{/key}
		</div>
		<div
			class="pointer-events-none absolute bottom-0 inset-x-0 z-10 h-8 -mt-8 bg-gradient-to-b from-transparent via-white/40 to-white/80 transition-opacity duration-200 {showBottomShadow
				? 'opacity-100'
				: 'opacity-0'}"
		></div>
		<div class="absolute bottom-0 right-0 z-20 m-2 pointer-events-auto">
			<div class="bg-black/50 text-white shadow-md flex items-center">
				<button
					class="p-1.5 hover:bg-white/10 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
					title="Zoom out"
					aria-label="Zoom out"
					disabled={!validatedQueries}
					onclick={(e) => {
						e.stopPropagation();
						adjustScale(1 / 1.1);
					}}
				>
					<ZoomOutIcon class="w-4 h-4" />
				</button>
				<div class="w-px h-4 bg-white/20"></div>
				<button
					class="p-1.5 hover:bg-white/10 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
					title="Zoom in"
					aria-label="Zoom in"
					disabled={!validatedQueries}
					onclick={(e) => {
						e.stopPropagation();
						adjustScale(1.1);
					}}
				>
					<ZoomInIcon class="w-4 h-4" />
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	/* Ensure the CodeMirror editor fills its container */
	:global(.cm-editor) {
		height: 100%;
	}
	:global(.cm-query-highlight-fuchsia) {
		background: color-mix(in srgb, var(--color-fuchsia-200) 50%, transparent);
		outline: 1px solid color-mix(in srgb, var(--color-fuchsia-400) 50%, transparent);
	}
	:global(.cm-query-highlight-teal) {
		background: color-mix(in srgb, var(--color-teal-200) 50%, transparent);
		outline: 1px solid color-mix(in srgb, var(--color-teal-400) 50%, transparent);
	}
	:global(.cm-query-highlight-amber) {
		background: color-mix(in srgb, var(--color-amber-200) 50%, transparent);
		outline: 1px solid color-mix(in srgb, var(--color-amber-400) 50%, transparent);
	}
</style>
