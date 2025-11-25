<script lang="ts">
	import type { ProfileEventData } from '$lib/train/protocol';
	import * as echarts from 'echarts';
	import { onMount } from 'svelte';

	interface Props {
		events: ProfileEventData[];
		minTime: number;
		maxTime: number;
		highlightedBufferId: string | null;
		onTensorHover: (bufferId: string | null, tensorIds: string[]) => void;
		onDataZoomChange: (start: number, end: number) => void;
		dataZoomStart: number;
		dataZoomEnd: number;
	}

	let {
		events,
		minTime,
		maxTime,
		highlightedBufferId,
		onTensorHover,
		onDataZoomChange,
		dataZoomStart,
		dataZoomEnd
	}: Props = $props();

	let chartContainer: HTMLDivElement;
	let chart: echarts.ECharts | null = null;
	let isUpdatingZoom = false;
	let lastEventsHash = '';

	// Represents a tensor's presence in memory at a point in time
	interface TensorState {
		tensorId: string;
		name: string;
		bufferId: string;
		shape: string;
		dtype: string;
		sizeMB: number;
		startTime: number;
		endTime: number;
	}

	// A point in the tensor's memory path (for drawing the "squiggly" shape)
	interface PathPoint {
		time: number;
		yBottom: number;
		yTop: number;
	}

	// The polygon data for rendering a tensor
	interface TensorPolygon {
		tensorId: string;
		name: string;
		bufferId: string;
		shape: string;
		dtype: string;
		sizeMB: number;
		path: PathPoint[];
	}

	function hashEvents(evts: ProfileEventData[]): string {
		// Quick hash to detect if events actually changed
		return `${evts.length}_${evts[0]?.start_us}_${evts[evts.length - 1]?.start_us}`;
	}

	function processEventsToPolygons(evts: ProfileEventData[]): {
		polygons: TensorPolygon[];
		peakMemory: number;
		tensorCount: number;
	} {
		// Extract tensor events
		const tensorEvents = evts.filter(
			(e) => e.category === 'tensor_allocation' || e.category === 'tensor_deallocation'
		);

		if (tensorEvents.length === 0) {
			return { polygons: [], peakMemory: 0, tensorCount: 0 };
		}

		// Build tensor lifecycle map
		const tensors = new Map<string, TensorState>();
		const maxTimeMs = maxTime;

		for (const event of tensorEvents) {
			const tensorId = event.metadata?.tensor_id || '';
			if (event.category === 'tensor_allocation') {
				const sizeBytes = parseInt(event.metadata?.size_bytes || '0', 10);
				tensors.set(tensorId, {
					tensorId,
					name: event.name,
					bufferId: event.metadata?.buffer_id || '',
					shape: event.metadata?.shape || '',
					dtype: event.metadata?.dtype || '',
					sizeMB: sizeBytes / (1024 * 1024),
					startTime: event.start_us / 1000,
					endTime: maxTimeMs // Will be updated if deallocated
				});
			} else if (event.category === 'tensor_deallocation') {
				const tensor = tensors.get(tensorId);
				if (tensor) {
					tensor.endTime = event.start_us / 1000;
				}
			}
		}

		// Collect all unique time points where memory layout changes
		const timePoints = new Set<number>();
		for (const tensor of tensors.values()) {
			timePoints.add(tensor.startTime);
			timePoints.add(tensor.endTime);
		}
		const sortedTimes = Array.from(timePoints).sort((a, b) => a - b);

		// At each time point, compute the stacked memory layout
		// Tensors are stacked in order of their start time (first allocated = bottom)
		const polygons: TensorPolygon[] = [];
		let peakMemory = 0;

		// Initialize polygon paths for each tensor
		for (const tensor of tensors.values()) {
			polygons.push({
				tensorId: tensor.tensorId,
				name: tensor.name,
				bufferId: tensor.bufferId,
				shape: tensor.shape,
				dtype: tensor.dtype,
				sizeMB: tensor.sizeMB,
				path: []
			});
		}

		// For each time point, calculate where each active tensor sits in the stack
		for (const time of sortedTimes) {
			// Get tensors active at this time, sorted by start time (oldest at bottom)
			const activeTensors = Array.from(tensors.values())
				.filter((t) => t.startTime <= time && t.endTime > time)
				.sort((a, b) => a.startTime - b.startTime);

			// Stack them
			let currentY = 0;
			const yPositions = new Map<string, { bottom: number; top: number }>();

			for (const tensor of activeTensors) {
				yPositions.set(tensor.tensorId, {
					bottom: currentY,
					top: currentY + tensor.sizeMB
				});
				currentY += tensor.sizeMB;
			}

			if (currentY > peakMemory) {
				peakMemory = currentY;
			}

			// Add path points for tensors that are active at this time
			for (const polygon of polygons) {
				const tensor = tensors.get(polygon.tensorId)!;

				// Only add points within the tensor's lifetime
				if (time >= tensor.startTime && time <= tensor.endTime) {
					const pos = yPositions.get(polygon.tensorId);
					if (pos) {
						// Add a point just before if this is the start
						if (polygon.path.length === 0 && time === tensor.startTime) {
							polygon.path.push({
								time: time,
								yBottom: pos.bottom,
								yTop: pos.top
							});
						} else if (polygon.path.length > 0) {
							// Check if position changed - if so, add transition points
							const lastPoint = polygon.path[polygon.path.length - 1];
							if (lastPoint.yBottom !== pos.bottom || lastPoint.yTop !== pos.top) {
								// Add point at previous position at current time (creates vertical edge)
								polygon.path.push({
									time: time,
									yBottom: lastPoint.yBottom,
									yTop: lastPoint.yTop
								});
							}
						}

						polygon.path.push({
							time: time,
							yBottom: pos.bottom,
							yTop: pos.top
						});
					}
				}
			}
		}

		// Close off paths for tensors that end
		for (const polygon of polygons) {
			const tensor = tensors.get(polygon.tensorId)!;
			if (polygon.path.length > 0) {
				const lastPoint = polygon.path[polygon.path.length - 1];
				// If the tensor ends at its endTime, make sure we have a closing point
				if (lastPoint.time < tensor.endTime) {
					polygon.path.push({
						time: tensor.endTime,
						yBottom: lastPoint.yBottom,
						yTop: lastPoint.yTop
					});
				}
			}
		}

		return { polygons, peakMemory, tensorCount: tensors.size };
	}

	function getColorForTensor(bufferId: string, isHighlighted: boolean): string {
		const hash = bufferId.split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
		const hue = Math.abs(hash) % 360;
		const saturation = isHighlighted ? 75 : 50;
		const lightness = isHighlighted ? 55 : 65;
		return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
	}

	function buildPolygonPoints(path: PathPoint[]): number[][] {
		if (path.length < 2) return [];

		// Build closed polygon: go forward along top edge, then backward along bottom edge
		const points: number[][] = [];

		// Top edge (forward)
		for (const p of path) {
			points.push([p.time, p.yTop]);
		}

		// Bottom edge (backward)
		for (let i = path.length - 1; i >= 0; i--) {
			points.push([path[i].time, path[i].yBottom]);
		}

		return points;
	}

	function updateChart() {
		if (!chart) return;

		const currentHash = hashEvents(events);
		if (currentHash === lastEventsHash && events.length > 0) {
			// Only update highlight styles, not full redraw
			return;
		}
		lastEventsHash = currentHash;

		if (events.length === 0) {
			chart.setOption({ series: [] }, { replaceMerge: ['series'] });
			return;
		}

		const { polygons, peakMemory, tensorCount } = processEventsToPolygons(events);

		if (polygons.length === 0 || peakMemory === 0) {
			chart.setOption({ series: [] }, { replaceMerge: ['series'] });
			return;
		}

		const timeRange = maxTime - minTime;
		const timePadding = timeRange * 0.02;

		// Create a series for each tensor polygon
		const series: echarts.SeriesOption[] = polygons
			.filter((p) => p.path.length >= 2)
			.map((polygon) => {
				const points = buildPolygonPoints(polygon.path);
				const isHighlighted = highlightedBufferId === polygon.bufferId;
				const isDimmed = highlightedBufferId && !isHighlighted;

				return {
					type: 'custom',
					renderItem: (
						params: echarts.CustomSeriesRenderItemParams,
						api: echarts.CustomSeriesRenderItemAPI
					) => {
						// Transform polygon points to pixel coordinates
						const pixelPoints = points.map((p) => api.coord(p) as [number, number]);

						return {
							type: 'polygon',
							shape: {
								points: pixelPoints
							},
							style: {
								fill: getColorForTensor(polygon.bufferId, isHighlighted),
								opacity: isDimmed ? 0.25 : 0.8,
								stroke: isHighlighted ? '#f97316' : 'rgba(255,255,255,0.3)',
								lineWidth: isHighlighted ? 2 : 0.5
							},
							emphasis: {
								style: {
									opacity: 1,
									stroke: '#f97316',
									lineWidth: 2
								}
							}
						};
					},
					data: [points], // Single data item containing all points
					encode: { x: 0, y: 1 },
					z: isHighlighted ? 100 : 10,
					// Store metadata for tooltip
					tensorId: polygon.tensorId,
					name: polygon.name,
					bufferId: polygon.bufferId,
					shape: polygon.shape,
					dtype: polygon.dtype,
					sizeMB: polygon.sizeMB,
					startTime: polygon.path[0]?.time,
					endTime: polygon.path[polygon.path.length - 1]?.time
				} as echarts.SeriesOption & {
					tensorId: string;
					bufferId: string;
					shape: string;
					dtype: string;
					sizeMB: number;
					startTime: number;
					endTime: number;
				};
			});

		const option: echarts.EChartsOption = {
			tooltip: {
				trigger: 'item',
				formatter: (params: unknown) => {
					const p = params as { seriesIndex?: number };
					if (p.seriesIndex === undefined) return '';
					const s = series[p.seriesIndex] as echarts.SeriesOption & {
						name: string;
						tensorId: string;
						bufferId: string;
						shape: string;
						dtype: string;
						sizeMB: number;
						startTime: number;
						endTime: number;
					};
					if (!s) return '';

					return `
						<div style="font-size: 12px;">
							<div style="font-weight: bold; margin-bottom: 4px; color: #10b981;">${s.name}</div>
							<table style="font-size: 11px;">
								<tr><td style="padding-right: 8px; color: #64748b;">Tensor ID:</td><td>${s.tensorId}</td></tr>
								<tr><td style="padding-right: 8px; color: #64748b;">Buffer ID:</td><td>${s.bufferId}</td></tr>
								<tr><td style="padding-right: 8px; color: #64748b;">Shape:</td><td>${s.shape}</td></tr>
								<tr><td style="padding-right: 8px; color: #64748b;">DType:</td><td>${s.dtype}</td></tr>
								<tr><td style="padding-right: 8px; color: #64748b;">Size:</td><td><strong>${s.sizeMB.toFixed(2)} MB</strong></td></tr>
								<tr><td style="padding-right: 8px; color: #64748b;">Start:</td><td>${s.startTime?.toFixed(2)} ms</td></tr>
								<tr><td style="padding-right: 8px; color: #64748b;">Duration:</td><td>${((s.endTime || 0) - (s.startTime || 0)).toFixed(2)} ms</td></tr>
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
				bottom: 30,
				left: 60,
				right: 20
			},
			dataZoom: [
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
				axisLabel: { show: false },
				axisLine: { show: true, lineStyle: { color: '#94a3b8' } },
				splitLine: { show: true, lineStyle: { color: '#f1f5f9' } }
			},
			yAxis: {
				type: 'value',
				min: 0,
				max: peakMemory * 1.05,
				name: 'Tensor Memory (MB)',
				nameLocation: 'middle',
				nameGap: 45,
				axisLine: { show: true, lineStyle: { color: '#94a3b8' } },
				splitLine: { show: true, lineStyle: { color: '#f1f5f9' } }
			},
			series
		};

		chart.setOption(option, { replaceMerge: ['series'] });
	}

	onMount(() => {
		chart = echarts.init(chartContainer);

		// Handle zoom changes
		chart.on('datazoom', (params: unknown) => {
			if (isUpdatingZoom) return;
			const p = params as { batch?: Array<{ start: number; end: number }> };
			if (p.batch && p.batch[0]) {
				onDataZoomChange(p.batch[0].start, p.batch[0].end);
			}
		});

		// Handle hover for cross-highlighting
		chart.on('mouseover', { seriesIndex: true }, (params: unknown) => {
			const p = params as { seriesIndex?: number };
			if (p.seriesIndex !== undefined) {
				const s = chart?.getOption()?.series as Array<{ bufferId?: string; tensorId?: string }>;
				if (s && s[p.seriesIndex]) {
					const bufferId = s[p.seriesIndex].bufferId;
					const tensorId = s[p.seriesIndex].tensorId;
					if (bufferId) {
						onTensorHover(bufferId, tensorId ? [tensorId] : []);
					}
				}
			}
		});

		chart.on('mouseout', () => {
			onTensorHover(null, []);
		});

		const resizeObserver = new ResizeObserver(() => {
			chart?.resize();
		});
		resizeObserver.observe(chartContainer);

		// Initial render
		updateChart();

		return () => {
			resizeObserver.disconnect();
			chart?.dispose();
		};
	});

	// Update chart when events change (but not on every render)
	$effect(() => {
		// Access dependencies
		const _ = [events.length, minTime, maxTime];
		if (chart) {
			updateChart();
		}
	});

	// Handle highlight changes separately to avoid full redraws
	$effect(() => {
		if (chart && highlightedBufferId !== undefined) {
			// Force redraw for highlight changes
			lastEventsHash = '';
			updateChart();
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

	const tensorCount = $derived.by(() => {
		const tensorEvents = events.filter((e) => e.category === 'tensor_allocation');
		return tensorEvents.length;
	});
</script>

<div class="w-full">
	<div class="text-xs text-slate-500 px-2 mb-1">Tensors ({tensorCount} allocations)</div>
	<div bind:this={chartContainer} class="w-full h-[180px]"></div>
</div>
