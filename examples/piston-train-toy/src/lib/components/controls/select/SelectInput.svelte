<script lang="ts">
	import type { Snippet } from 'svelte';

	import Portal from 'svelte-portal';

	import ChevronIcon from '../../ChevronIcon.svelte';
	import FormLabel from '../FormLabel.svelte';
	import ResetValueButton from '../ResetValueButton.svelte';

	export type SelectOption<T extends object = Record<string, unknown>> = {
		value: string | number;
		disabled?: boolean;
	} & T;

	type Option = SelectOption;
	type Group = { label: string; options: SelectOption[] };
	type $$Props = {
		label?: string;
		value: string | number;
		options?: SelectOption[];
		groups?: Group[];
		id: string;
		class?: string;
		hasDefaultValue?: boolean;
		onReset?: () => void;
		trigger?: Snippet<[selected: unknown | undefined]>;
		option?: Snippet<[option: unknown, selected: boolean, index: number]>;
	};

	let {
		label,
		value = $bindable(),
		options = [],
		groups = undefined,
		id,
		class: wrapperClass = '',
		hasDefaultValue = false,
		onReset = undefined,
		trigger,
		option: optionSnippet
	}: $$Props = $props();

	let isOpen = $state(false);
	let highlightedIndex = $state(-1);
	let wrapperEl = $state<HTMLDivElement | null>(null);
	let triggerEl = $state<HTMLButtonElement | null>(null);
	let menuEl = $state<HTMLUListElement | null>(null);
	let placement = $state<'bottom' | 'top'>('bottom');
	let menuStyle = $state('');

	// Derived option sources (supports groups)
	const hasGroups = $derived(Boolean(groups && groups.length > 0));
	const flatOptions = $derived(hasGroups ? (groups as Group[]).flatMap((g) => g.options) : options);

	type MenuGroup = { kind: 'group'; id: string; label: string };
	type MenuOpt = { kind: 'option'; id: string; option: Option; optIndex: number };
	type MenuItem = MenuGroup | MenuOpt;

	function buildMenuItems(): MenuItem[] {
		if (!hasGroups) {
			return flatOptions.map((opt, i) => ({
				kind: 'option',
				id: String(i),
				option: opt,
				optIndex: i
			}));
		}
		const out: MenuItem[] = [];
		let offset = 0;
		for (let gi = 0; gi < (groups as Group[]).length; gi++) {
			const g = (groups as Group[])[gi]!;
			out.push({ kind: 'group', id: `g${gi}`, label: g.label });
			for (let i = 0; i < g.options.length; i++) {
				const opt = g.options[i]!;
				out.push({ kind: 'option', id: `g${gi}o${i}`, option: opt, optIndex: offset + i });
			}
			offset += g.options.length;
		}
		return out;
	}

	const menuItems = $derived(buildMenuItems());

	const selectedIndex = $derived(flatOptions.findIndex((o) => o.value === value));
	const selectedOption = $derived(selectedIndex >= 0 ? flatOptions[selectedIndex] : undefined);

	function getOptionText(opt: Option): string {
		return String(
			(opt as Record<string, unknown>)?.text ?? (opt as Record<string, unknown>)?.label ?? opt.value
		);
	}

	function isOptionDisabled(opt: Option): boolean {
		return (opt as Record<string, unknown>)?.disabled === true;
	}

	function firstEnabledIndex(): number {
		for (let i = 0; i < menuItems.length; i++) {
			const item = menuItems[i];
			if (item.kind === 'option' && !isOptionDisabled(item.option)) return i;
		}
		return -1;
	}

	function nextEnabledIndex(fromIndex: number, direction: 1 | -1): number {
		let i = fromIndex;
		while (true) {
			i += direction;
			if (i < 0 || i >= menuItems.length) return fromIndex;
			const item = menuItems[i];
			if (item.kind === 'option' && !isOptionDisabled(item.option)) return i;
		}
	}

	function openMenu() {
		if (!isOpen) {
			isOpen = true;
			highlightedIndex =
				selectedIndex >= 0 && selectedOption && !isOptionDisabled(selectedOption)
					? menuItems.findIndex((it) => it.kind === 'option' && it.optIndex === selectedIndex)
					: firstEnabledIndex();
			queueMicrotask(() => reposition());
		}
	}

	function closeMenu() {
		if (isOpen) {
			isOpen = false;
			highlightedIndex = -1;
		}
	}

	function toggleMenu() {
		if (isOpen) {
			closeMenu();
		} else {
			openMenu();
		}
	}

	function reposition() {
		if (!triggerEl) return;
		const rect = triggerEl.getBoundingClientRect();
		const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
		const spaceBelow = viewportHeight - rect.bottom;
		const spaceAbove = rect.top;
		const measured = menuEl?.getBoundingClientRect().height;
		const estimatedMenuHeight = measured && measured > 0 ? measured : 240; // ~max-h-60
		placement = spaceBelow < estimatedMenuHeight && spaceAbove > spaceBelow ? 'top' : 'bottom';

		const menuHeight = measured && measured > 0 ? measured : estimatedMenuHeight;
		const top = placement === 'bottom' ? rect.bottom : rect.top - menuHeight;
		menuStyle = `left:0;top:0;transform:translate(${rect.left}px,${Math.max(0, top)}px);width:${rect.width}px;`;
	}

	function selectByMenuIndex(menuIndex: number) {
		const item = menuItems[menuIndex];
		if (!item || item.kind !== 'option') return;
		selectIndex(item.optIndex);
	}

	function selectIndex(index: number) {
		const next = flatOptions[index];
		if (!next || isOptionDisabled(next)) return;
		value = next.value;
		closeMenu();
		queueMicrotask(() => triggerEl?.focus());
	}

	function onTriggerKeydown(e: KeyboardEvent) {
		switch (e.key) {
			case 'ArrowDown':
				e.preventDefault();
				if (!isOpen) openMenu();
				else {
					const base =
						highlightedIndex < 0
							? selectedIndex >= 0
								? menuItems.findIndex((it) => it.kind === 'option' && it.optIndex === selectedIndex)
								: -1
							: highlightedIndex;
					if (base < 0) {
						highlightedIndex = firstEnabledIndex();
					} else {
						highlightedIndex = nextEnabledIndex(base, 1);
					}
				}
				break;
			case 'ArrowUp':
				e.preventDefault();
				if (!isOpen) openMenu();
				else {
					const base =
						highlightedIndex < 0
							? selectedIndex >= 0
								? menuItems.findIndex((it) => it.kind === 'option' && it.optIndex === selectedIndex)
								: menuItems.length
							: highlightedIndex;
					if (base >= menuItems.length) {
						highlightedIndex = firstEnabledIndex();
					} else {
						highlightedIndex = nextEnabledIndex(base, -1);
					}
				}
				break;
			case 'Enter':
			case ' ':
				e.preventDefault();
				if (!isOpen) openMenu();
				else if (highlightedIndex >= 0) selectByMenuIndex(highlightedIndex);
				break;
			case 'Escape':
				if (isOpen) {
					e.preventDefault();
					closeMenu();
				}
				break;
		}
	}

	function onMenuKeydown(e: KeyboardEvent) {
		switch (e.key) {
			case 'ArrowDown':
				e.preventDefault();
				if (highlightedIndex < 0) {
					highlightedIndex = firstEnabledIndex();
				} else {
					highlightedIndex = nextEnabledIndex(highlightedIndex, 1);
				}
				break;
			case 'ArrowUp':
				e.preventDefault();
				if (highlightedIndex < 0) {
					highlightedIndex = firstEnabledIndex();
				} else {
					highlightedIndex = nextEnabledIndex(highlightedIndex, -1);
				}
				break;
			case 'Enter':
				e.preventDefault();
				if (highlightedIndex >= 0) selectByMenuIndex(highlightedIndex);
				break;
			case 'Escape':
				e.preventDefault();
				closeMenu();
				break;
		}
	}

	$effect(() => {
		if (!isOpen) return;
		const onDocPointerDown = (ev: PointerEvent) => {
			const target = ev.target as Node;
			if (!wrapperEl?.contains(target) && !(menuEl && menuEl.contains(target))) {
				closeMenu();
			}
		};
		const onResize = () => reposition();
		const onScroll = () => reposition();
		document.addEventListener('pointerdown', onDocPointerDown, true);
		window.addEventListener('resize', onResize);
		window.addEventListener('scroll', onScroll, true);
		reposition();
		return () => {
			document.removeEventListener('pointerdown', onDocPointerDown, true);
			window.removeEventListener('resize', onResize);
			window.removeEventListener('scroll', onScroll, true);
		};
	});
</script>

<div class="relative {wrapperClass}" bind:this={wrapperEl}>
	{#if label}
		<FormLabel forInputId={id} value={label} />
	{/if}
	<div class="flex gap-1.5 items-center">
		<div class="relative flex-1">
			<button
				{id}
				bind:this={triggerEl}
				type="button"
				aria-haspopup="listbox"
				aria-expanded={isOpen}
				aria-controls={`${id}-menu`}
				onclick={toggleMenu}
				onkeydown={onTriggerKeydown}
				class="block w-full border border-neutral-300 py-0.5 pl-1 pr-6 text-base focus:border-blue-500 focus:outline-none focus:ring-blue-500 px-2 text-left rounded-none appearance-none bg-white"
				class:mt-1={label}
			>
				{#if trigger}
					{@render trigger(selectedOption)}
				{:else}
					{selectedOption ? getOptionText(selectedOption) : 'Select...'}
				{/if}
			</button>
			<div
				class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-1 text-neutral-400"
			>
				<ChevronIcon direction="down" size={10} />
			</div>

			{#if isOpen}
				<Portal target="body">
					<ul
						bind:this={menuEl}
						id={`${id}-menu`}
						role="listbox"
						aria-activedescendant={isOpen &&
						highlightedIndex >= 0 &&
						menuItems[highlightedIndex]?.kind === 'option'
							? `${id}-opt-${highlightedIndex}`
							: undefined}
						tabindex="-1"
						onkeydown={onMenuKeydown}
						class="fixed z-[9999] border border-neutral-300 bg-white max-h-120 overflow-auto text-base rounded-none shadow-lg"
						style={menuStyle}
					>
						{#each menuItems as item, i (item.id)}
							{#if item.kind === 'group'}
								<li
									role="group"
									aria-label={item.label}
									class="px-1 py-0.5 text-2xs font-mono uppercase font-bold tracking-wider text-neutral-700 select-none border-neutral-300 bg-neutral-50"
									class:border-t={i > 0}
									class:border-b={i < menuItems.length - 1}
								>
									{item.label}
								</li>
							{:else}
								{@const opt = item.option}
								{@const isDisabled = isOptionDisabled(opt)}
								<li
									id={`${id}-opt-${i}`}
									role="option"
									aria-selected={opt.value === value}
									aria-disabled={isDisabled}
									class="px-0 py-0 select-none"
									class:bg-neutral-100={i === highlightedIndex && !isDisabled}
									onpointerenter={() => (!isDisabled ? (highlightedIndex = i) : undefined)}
								>
									<button
										type="button"
										class="block w-full text-left px-1 py-1 rounded-none"
										class:cursor-pointer={!isDisabled}
										class:cursor-not-allowed={isDisabled}
										class:opacity-60={isDisabled}
										disabled={isDisabled}
										onclick={() => selectByMenuIndex(i)}
									>
										{#if optionSnippet}
											{@render optionSnippet(opt, opt.value === value, item.optIndex)}
										{:else}
											{getOptionText(opt)}
										{/if}
									</button>
								</li>
							{/if}
						{/each}
					</ul>
				</Portal>
			{/if}
		</div>
		{#if onReset}
			<ResetValueButton {hasDefaultValue} {onReset} />
		{/if}
	</div>
</div>
