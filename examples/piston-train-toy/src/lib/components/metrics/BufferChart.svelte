<script lang="ts">
	import type { ProfileEventData } from '$lib/train/protocol';
	import * as echarts from 'echarts';
	import { onMount } from 'svelte';

	interface Props {
		events: ProfileEventData[];
		minTime: number;
		maxTime: number;
		highlightedBufferId: string | null;
		highlightedTensorIds: string[];
		onBufferHover: (bufferId: string | null, tensorIds: string[]) => void;
		onDataZoomChange: (start: number, end: number) => void;
		dataZoomStart: number;
		dataZoomEnd: number;
	}

	let {
		events,
		minTime,
		maxTime,
		highlightedBufferId,
		highlightedTensorIds,
		onBufferHover,
		onDataZoomChange,
		dataZoomStart,
		dataZoomEnd
	}: Props = $props();

	let chartContainer: HTMLDivElement;
	let chart: echarts.ECharts | null = null;
	let isUpdatingZoom = false;

	// Cache for processed data
	let lastEventsLength = 0;
	let cachedBlocks: AllocationBlock[] = [];
	let cachedPeakMemory = 0;

	// Track which tensors used which buffers
	let bufferToTensors = new Map<string, string[]>();

	interface AllocationBlock {
		name: string;
		category: string;
		bufferId: string;
		size: number;
		start: number;
		end: number;
		addr: number;
		metadata?: Record<string, string>;
	}

	function processEventsToBlocks(evts: ProfileEventData[]): {
		blocks: AllocationBlock[];
		peakMemory: number;
	} {
		// Build buffer-to-tensors mapping from tensor events
		bufferToTensors.clear();
		for (const event of evts) {
			if (event.category === 'tensor_allocation') {
				const bufferId = event.metadata?.buffer_id || '';
				const tensorId = event.metadata?.tensor_id || '';
				if (bufferId && tensorId) {
					if (!bufferToTensors.has(bufferId)) {
						bufferToTensors.set(bufferId, []);
					}
					bufferToTensors.get(bufferId)!.push(tensorId);
				}
			}
		}

		const allocations: Map<string, AllocationBlock> = new Map();
		const activeBlocks: Array<{
			key: string;
			startAddr: number;
			endAddr: number;
			sizeBytes: number;
			bufferId: string;
		}> = [];

		let peakMemory = 0;
		const sortedEvents = [...evts].sort((a, b) => a.start_us - b.start_us);
		const maxTimeMs = Math.max(...evts.map((e) => (e.start_us + e.duration_us) / 1000));

		for (const event of sortedEvents) {
			if (event.category === 'allocation') {
				const sizeBytes = parseInt(event.metadata?.size_bytes || '0', 10);
				const sizeMB = sizeBytes / (1024 * 1024);
				const startMs = event.start_us / 1000;
				const bufferId = event.metadata?.buffer_id || '';

				activeBlocks.sort((a, b) => a.startAddr - b.startAddr);

				let bestAddr = 0;
				if (activeBlocks.length > 0) {
					if (activeBlocks[0].startAddr >= sizeMB) {
						bestAddr = 0;
					} else {
						let foundGap = false;
						for (let i = 0; i < activeBlocks.length - 1; i++) {
							const gap = activeBlocks[i + 1].startAddr - activeBlocks[i].endAddr;
							if (gap >= sizeMB) {
								bestAddr = activeBlocks[i].endAddr;
								foundGap = true;
								break;
							}
						}
						if (!foundGap) {
							bestAddr = activeBlocks[activeBlocks.length - 1].endAddr;
						}
					}
				}

				const endAddr = bestAddr + sizeMB;
				if (endAddr > peakMemory) peakMemory = endAddr;

				const key = `alloc_${bufferId}_${startMs}`;
				const block: AllocationBlock = {
					name: event.name,
					category: event.metadata?.usage || 'buffer',
					bufferId,
					size: sizeMB,
					start: startMs,
					end: maxTimeMs,
					addr: bestAddr,
					metadata: event.metadata
				};

				allocations.set(key, block);
				activeBlocks.push({ key, startAddr: bestAddr, endAddr, sizeBytes, bufferId });
			} else if (event.category === 'deallocation') {
				const deallocTime = event.start_us / 1000;
				const deallocBufferId = event.metadata?.buffer_id || '';

				const matchIdx = activeBlocks.findIndex((b) => b.bufferId === deallocBufferId);
				if (matchIdx !== -1) {
					const match = activeBlocks[matchIdx];
					const alloc = allocations.get(match.key);
					if (alloc) alloc.end = deallocTime;
					activeBlocks.splice(matchIdx, 1);
				}
			}
		}

		// Add kernel events
		for (const event of sortedEvents) {
			if (event.category === 'kernel' && event.duration_us > 0) {
				const startMs = event.start_us / 1000;
				const durationMs = event.duration_us / 1000;
				allocations.set('kernel_' + event.name + '_' + startMs, {
					name: event.name,
					category: 'kernel',
					bufferId: '',
					size: 5,
					start: startMs,
					end: startMs + durationMs,
					addr: peakMemory + 10,
					metadata: event.metadata
				});
			}
		}

		// Add scope events
		for (const event of sortedEvents) {
			if (event.category === 'scope' && event.duration_us > 0) {
				const startMs = event.start_us / 1000;
				const durationMs = event.duration_us / 1000;
				allocations.set('scope_' + event.name + '_' + startMs, {
					name: event.name,
					category: 'scope',
					bufferId: '',
					size: 3,
					start: startMs,
					end: startMs + durationMs,
					addr: peakMemory + 20 + (event.stack?.length || 0) * 5
				});
			}
		}

		return { blocks: Array.from(allocations.values()), peakMemory };
	}

	function getColorForCategory(category: string, isHighlighted: boolean): string {
		if (isHighlighted) return '#f97316';
		switch (category) {
			case 'kernel':
				return '#60a5fa';
			case 'scope':
				return '#a78bfa';
			case 'STORAGE':
				return '#34d399';
			case 'COPY_DST':
			case 'COPY_SRC':
				return '#fbbf24';
			default:
				return '#94a3b8';
		}
	}

	function renderItem(
		params: echarts.CustomSeriesRenderItemParams,
		api: echarts.CustomSeriesRenderItemAPI
	): echarts.CustomSeriesRenderItemReturn {
		const start = api.coord([api.value(1), api.value(3)]);
		const end = api.coord([api.value(2), (api.value(3) as number) + (api.value(4) as number)]);
		const style = api.style();

		return {
			type: 'rect',
			shape: {
				x: start[0],
				y: end[1],
				width: Math.max(end[0] - start[0], 2),
				height: Math.max(start[1] - end[1], 2)
			},
			style,
			emphasis: {
				style: {
					...style,
					opacity: 1,
					stroke: '#f97316',
					lineWidth: 2
				}
			}
		};
	}

	function updateChart(forceRebuild = false) {
		if (!chart) return;

		// Check if we need to rebuild the data
		if (forceRebuild || events.length !== lastEventsLength) {
			lastEventsLength = events.length;
			if (events.length === 0) {
				cachedBlocks = [];
				cachedPeakMemory = 0;
			} else {
				const result = processEventsToBlocks(events);
				cachedBlocks = result.blocks;
				cachedPeakMemory = result.peakMemory;
			}
		}

		if (cachedBlocks.length === 0) {
			chart.setOption({ series: [{ data: [] }] });
			return;
		}

		const timeRange = maxTime - minTime;
		const timePadding = timeRange * 0.02;

		// Convert to ECharts data format
		const rawData = cachedBlocks.map((d, i) => [
			i,
			d.start,
			d.end,
			d.addr,
			d.size,
			d.name,
			d.category,
			d.bufferId
		]);

		const option: echarts.EChartsOption = {
			animation: false, // Disable animation to prevent repainting issues
			tooltip: {
				formatter: (params: unknown) => {
					const p = params as { data: { value: (string | number)[] } };
					const d = p.data?.value || [];
					if (!d || d.length < 8) return '';

					const bufferId = d[7] as string;
					const tensorsList = bufferToTensors.get(bufferId) || [];

					return `
						<div style="font-size: 12px;">
							<div style="font-weight: bold; margin-bottom: 4px; color: #6366f1;">${d[5]}</div>
							<table style="font-size: 11px;">
								<tr><td style="padding-right: 8px; color: #64748b;">Category:</td><td>${d[6]}</td></tr>
								${bufferId ? `<tr><td style="padding-right: 8px; color: #64748b;">Buffer ID:</td><td>${bufferId}</td></tr>` : ''}
								<tr><td style="padding-right: 8px; color: #64748b;">Start:</td><td>${(d[1] as number).toFixed(2)} ms</td></tr>
								<tr><td style="padding-right: 8px; color: #64748b;">Duration:</td><td>${((d[2] as number) - (d[1] as number)).toFixed(2)} ms</td></tr>
								<tr><td style="padding-right: 8px; color: #64748b;">Size:</td><td><strong>${(d[4] as number).toFixed(2)} MB</strong></td></tr>
								${tensorsList.length > 0 ? `<tr><td style="padding-right: 8px; color: #64748b;">Tensors:</td><td>${tensorsList.length} used this buffer</td></tr>` : ''}
							</table>
						</div>
					`;
				},
				backgroundColor: 'rgba(255, 255, 255, 0.95)',
				borderColor: '#e2e8f0',
				borderWidth: 1,
				textStyle: { color: '#1e293b' },
				padding: 12
			},
			grid: {
				top: 10,
				bottom: 60,
				left: 60,
				right: 20
			},
			dataZoom: [
				{
					type: 'slider',
					show: true,
					xAxisIndex: [0],
					bottom: 10,
					start: dataZoomStart,
					end: dataZoomEnd,
					height: 20,
					handleSize: '80%'
				},
				{
					type: 'inside',
					xAxisIndex: [0],
					filterMode: 'none',
					start: dataZoomStart,
					end: dataZoomEnd
				}
			],
			xAxis: {
				type: 'value',
				min: minTime - timePadding,
				max: maxTime + timePadding,
				name: 'Time (ms)',
				nameLocation: 'middle',
				nameGap: 25,
				axisLine: { show: true, lineStyle: { color: '#94a3b8' } },
				splitLine: { show: true, lineStyle: { color: '#f1f5f9' } }
			},
			yAxis: {
				type: 'value',
				min: 0,
				max: cachedPeakMemory * 1.1,
				name: 'Buffer Memory (MB)',
				nameLocation: 'middle',
				nameGap: 45,
				axisLine: { show: true, lineStyle: { color: '#94a3b8' } },
				splitLine: { show: true, lineStyle: { color: '#f1f5f9' } }
			},
			series: [
				{
					type: 'custom',
					renderItem: renderItem,
					encode: {
						x: [1, 2],
						y: [3, 4]
					},
					data: rawData.map((item) => {
						const bufferId = item[7] as string;
						const category = item[6] as string;
						const tensorsForBuffer = bufferToTensors.get(bufferId) || [];
						const isHighlightedByTensor = highlightedTensorIds.some((tid) =>
							tensorsForBuffer.includes(tid)
						);
						const isHighlighted =
							(highlightedBufferId && highlightedBufferId === bufferId) || isHighlightedByTensor;
						const isDimmed = highlightedBufferId && !isHighlighted && bufferId;

						return {
							value: item,
							itemStyle: {
								color: getColorForCategory(category, isHighlighted),
								opacity: isDimmed ? 0.25 : 0.8,
								borderColor: isHighlighted ? '#f97316' : 'rgba(255,255,255,0.3)',
								borderWidth: isHighlighted ? 2 : 0.5
							}
						};
					})
				}
			]
		};

		chart.setOption(option);
	}

	onMount(() => {
		chart = echarts.init(chartContainer);

		chart.on('datazoom', (params: unknown) => {
			if (isUpdatingZoom) return;
			const p = params as {
				batch?: Array<{ start: number; end: number }>;
				start?: number;
				end?: number;
			};
			if (p.batch && p.batch[0]) {
				onDataZoomChange(p.batch[0].start, p.batch[0].end);
			} else if (p.start !== undefined && p.end !== undefined) {
				onDataZoomChange(p.start, p.end);
			}
		});

		chart.on('mouseover', (params: unknown) => {
			const p = params as { data?: { value: (string | number)[] } };
			if (p.data?.value) {
				const bufferId = p.data.value[7] as string;
				if (bufferId) {
					const tensorIds = bufferToTensors.get(bufferId) || [];
					onBufferHover(bufferId, tensorIds);
				}
			}
		});

		chart.on('mouseout', () => {
			onBufferHover(null, []);
		});

		const resizeObserver = new ResizeObserver(() => {
			chart?.resize();
		});
		resizeObserver.observe(chartContainer);

		// Initial render
		updateChart(true);

		return () => {
			resizeObserver.disconnect();
			chart?.dispose();
		};
	});

	// Update chart when events change
	$effect(() => {
		const _ = [events.length, minTime, maxTime];
		if (chart) {
			updateChart(true);
		}
	});

	// Update styles when highlights change (without rebuilding data)
	$effect(() => {
		const _ = [highlightedBufferId, highlightedTensorIds];
		if (chart && cachedBlocks.length > 0) {
			updateChart(false);
		}
	});

	// Sync dataZoom from parent
	$effect(() => {
		if (chart && !isUpdatingZoom) {
			isUpdatingZoom = true;
			chart.dispatchAction({
				type: 'dataZoom',
				start: dataZoomStart,
				end: dataZoomEnd
			});
			isUpdatingZoom = false;
		}
	});

	const bufferCount = $derived(cachedBlocks.filter((b) => b.bufferId).length);
</script>

<div class="w-full">
	<div class="flex items-center justify-between mb-1 px-2">
		<div class="text-xs text-slate-500">Buffers ({bufferCount} allocations)</div>
		<div class="flex gap-3 text-xs">
			<div class="flex items-center gap-1">
				<span class="w-2.5 h-2.5 bg-emerald-400 rounded-sm"></span>
				<span class="text-slate-500">Storage</span>
			</div>
			<div class="flex items-center gap-1">
				<span class="w-2.5 h-2.5 bg-blue-400 rounded-sm"></span>
				<span class="text-slate-500">Kernel</span>
			</div>
			<div class="flex items-center gap-1">
				<span class="w-2.5 h-2.5 bg-purple-400 rounded-sm"></span>
				<span class="text-slate-500">Scope</span>
			</div>
		</div>
	</div>
	<div bind:this={chartContainer} class="w-full h-[250px]"></div>
</div>
